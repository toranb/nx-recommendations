defmodule Example.Keywords do
  import Nx.Defn

  @moduledoc false

  @model "google-bert/bert-base-uncased"
  @max_length 38

  NimbleCSV.define(KeywordParser, separator: ",", escape: "\"")

  def go() do
    num_epochs = 12
    learning_rate = 0.02
    model = train(num_epochs, learning_rate)
    model_params = %{model: model}
    serialized_container = Nx.serialize(model_params)
    File.write!("#{Path.dirname(__ENV__.file)}/keyword_data", serialized_container)

    :ok
  end

  def training_data() do
    "keywords.csv"
    |> File.stream!()
    |> KeywordParser.parse_stream(skip_headers: false)
    |> Stream.map(fn [query, keywords] ->
      {query, keywords}
    end)
    |> Enum.to_list()
  end

  defp one_hot(query_tokens, keywords, tokenizer) do
    initial_labels = List.duplicate(0, length(query_tokens))

    Enum.reduce(keywords, initial_labels, fn _keyword, acc ->
      keyword_inputs = Bumblebee.apply_tokenizer(tokenizer, keywords)

      keyword_tokens =
        keyword_inputs["input_ids"]
        |> Nx.to_flat_list()
        |> Enum.map(&Bumblebee.Tokenizer.decode(tokenizer, [&1]))

      mark_keyword_in_query(acc, query_tokens, keyword_tokens)
    end)
  end

  defp mark_keyword_in_query(labels, query_tokens, keyword_tokens) do
    keyword_tokens = Enum.reject(keyword_tokens, &(&1 == ""))

    Enum.reduce(0..(length(query_tokens) - 1), labels, fn i, acc ->
      current_token = Enum.at(query_tokens, i)

      cond do
        current_token == "##s" and i > 0 ->
          prev_token = Enum.at(query_tokens, i - 1)

          if Enum.member?(keyword_tokens, prev_token) do
            List.update_at(acc, i, fn _ -> 1 end)
          else
            acc
          end

        String.starts_with?(current_token, "##") ->
          prev_token = if i > 0, do: Enum.at(query_tokens, i - 1), else: nil

          if prev_token && Enum.member?(keyword_tokens, prev_token) do
            List.update_at(acc, i, fn _ -> 1 end)
          else
            acc
          end

        true ->
          if Enum.member?(keyword_tokens, current_token) do
            List.update_at(acc, i, fn _ -> 1 end)
          else
            acc
          end
      end
    end)
  end

  def prepare_data(examples, model_info, tokenizer) do
    tokenizer =
      Bumblebee.configure(tokenizer,
        pad_direction: :right,
        return_token_type_ids: false,
        return_length: false
      )

    {:ok, config} = Bumblebee.load_generation_config({:hf, @model})
    generation_config = Bumblebee.configure(config, max_new_tokens: @max_length)
    serving = Example.Generation.get_text(model_info, tokenizer, generation_config, compile: [batch_size: 512, sequence_length: [@max_length]], defn_options: [compiler: EXLA])

    texts = examples |> Enum.map(fn {text, _} -> text end)
    %{hidden_states: embeddings, tokens: tokens} = Nx.Serving.run(serving, texts)

    0..(elem(Nx.shape(embeddings), 0) - 1)
    |> Enum.map(fn i ->
      {_, keywords} = Enum.at(examples, i)
      embeddings = Nx.slice(embeddings, [i, 0, 0], [1, @max_length, 768]) |> Nx.squeeze(axes: [0])
      tokens = Nx.slice(tokens, [i, 0], [1, @max_length]) |> Nx.to_flat_list()
      token_strings = tokens |> Enum.map(&Bumblebee.Tokenizer.decode(tokenizer, [&1]))

      keyword_strings =
        keywords
        |> String.downcase()
        |> String.split()
        |> Enum.map(&String.trim/1)

      labels = one_hot(token_strings, keyword_strings, tokenizer)

      {token_strings, embeddings, Nx.tensor(labels)}
    end)
  end

  def prepare_data_slower(examples, model_info, tokenizer) do
    %{model: model, params: params} = model_info

    examples
    |> Enum.map(fn {text, keywords} ->
      inputs = Bumblebee.apply_tokenizer(tokenizer, [text])
      %{hidden_state: hidden_state} = Axon.predict(model, params, inputs)
      embeddings = Nx.squeeze(hidden_state, axes: [0])

      token_strings =
        inputs["input_ids"]
        |> Nx.to_flat_list()
        |> Enum.map(&Bumblebee.Tokenizer.decode(tokenizer, [&1]))

      keyword_strings =
        keywords
        |> String.downcase()
        |> String.split()
        |> Enum.map(&String.trim/1)

      labels = one_hot(token_strings, keyword_strings, tokenizer)

      {token_strings, embeddings, Nx.tensor(labels)}
    end)
  end

  defn relu(x) do
    custom_grad(
      Nx.max(x, 0),
      [x],
      fn g -> [Nx.select(Nx.greater(x, 0), g, Nx.broadcast(0, g))] end
    )
  end

  defn initialize_dense(opts \\ []) do
    input_dim = opts[:input_dim]
    hidden_dim = opts[:hidden_dim]
    key_one = Nx.Random.key(42)
    key_two = Nx.Random.key(23)
    w1 = Nx.Random.normal_split(key_one, 0.0, 0.1, shape: {input_dim, hidden_dim}, type: :f16)
    b1 = Nx.broadcast(0.0, {hidden_dim})
    w2 = Nx.Random.normal_split(key_two, 0.0, 0.1, shape: {hidden_dim, 1}, type: :f16)
    b2 = Nx.broadcast(0.0, {1})
    {w1, b1, w2, b2}
  end

  defn forward({w1, b1, w2, b2}, embeddings) do
    embeddings
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> relu()
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> Nx.sigmoid()
  end

  defn loss(params, embeddings, labels) do
    max_value = 1.0e6
    probs = forward(params, embeddings)
    positive_term = labels * Nx.log(probs + 1.0e-10)
    negative_term = (1 - labels) * Nx.log(1 - probs + 1.0e-10)
    loss = -Nx.mean(positive_term + negative_term)
    Nx.select(Nx.is_infinity(loss) or Nx.is_nan(loss), max_value, loss)
  end

  defn update({w1, b1, w2, b2} = params, embeddings, labels, learning_rate) do
    {loss_value, {grad_w1, grad_b1, grad_w2, grad_b2}} =
      value_and_grad(params, &loss(&1, embeddings, labels))

    updated_w1 = w1 - grad_w1 * learning_rate
    updated_b1 = b1 - grad_b1 * learning_rate
    updated_w2 = w2 - grad_w2 * learning_rate
    updated_b2 = b2 - grad_b2 * learning_rate

    {{updated_w1, updated_b1, updated_w2, updated_b2}, loss_value}
  end

  def train(num_epochs, learning_rate) do
    {:ok, model_info} = Bumblebee.load_model({:hf, @model}, architecture: :base)
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model})

    bert_vocab = 768
    hidden_dim = 128
    examples = training_data()

    preprocessed_examples =
      prepare_data(examples, model_info, tokenizer)
      |> Enum.map(fn {_, e, l} ->
        {e, l}
      end)

    initial_params = initialize_dense(input_dim: bert_vocab, hidden_dim: hidden_dim)

    Enum.reduce(1..num_epochs, initial_params, fn epoch, epoch_params ->
      shuffled_examples = Enum.shuffle(preprocessed_examples)
      batches = Enum.chunk_every(shuffled_examples, 16)

      {final_epoch_params, total_loss} =
        Enum.reduce(batches, {epoch_params, 0.0}, fn batch, {current_params, acc_loss} ->

        {batch_params, batch_loss} =
          Enum.reduce(batch, {current_params, 0.0}, fn {embeddings, target}, {params, loss} ->
            target_shape = elem(Nx.shape(target), 0)
            labels = Nx.reshape(target, {target_shape, :auto})
            {new_params, example_loss} = update(params, embeddings, labels, learning_rate)
            {new_params, loss + Nx.to_number(example_loss)}
          end)

          # deallocate intermediate params
          if current_params != epoch_params do
            Nx.backend_deallocate(current_params)
          end

          avg_batch_loss = batch_loss / length(batch)
          {batch_params, acc_loss + avg_batch_loss}
        end)

      avg_epoch_loss = total_loss / length(batches)
      IO.puts("Epoch #{epoch}, Avg Loss: #{avg_epoch_loss}")

      final_epoch_params
    end)
  end

  def eval_data() do
    "eval.csv"
    |> File.stream!()
    |> KeywordParser.parse_stream(skip_headers: false)
    |> Stream.map(fn [query, keywords] ->
      {query, keywords}
    end)
    |> Enum.to_list()
  end

  def evaluate() do
    evals = eval_data()

    model_data = File.read!("#{Path.dirname(__ENV__.file)}/keyword_data")
    deserialized_data = Nx.deserialize(model_data)
    params = deserialized_data.model

    {:ok, model_info} = Bumblebee.load_model({:hf, @model}, architecture: :base)
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model})

    evals
    |> Enum.map(fn {query, keywords} ->
      examples = [{query, keywords}]
      result = eval(examples, params, model_info, tokenizer)

      sorted_result = result |> String.trim() |> String.split() |> Enum.sort()
      sorted_keywords = keywords |> String.trim() |> String.split() |> Enum.sort()
      equals = sorted_result == sorted_keywords

      result_words = result |> String.split()
      keyword_words = keywords |> String.split()

      missing_words = keyword_words -- result_words
      extra_words = result_words -- keyword_words

      %{
        equals: equals,
        query: query,
        result: result,
        keywords: keywords,
        missing: missing_words,
        extra: extra_words,
        difference_score: length(missing_words) + length(extra_words)
      }
    end)
    |> Enum.reject(& &1.equals)
    # |> IO.inspect(label: "eval details")

    {correct_count, total_count} =
      Enum.map(evals, fn {query, keywords} ->
        examples = [{query, keywords}]
        result = eval(examples, params, model_info, tokenizer)
        sorted_result = result |> String.trim() |> String.split() |> Enum.sort()
        sorted_keywords = keywords |> String.trim() |> String.split() |> Enum.sort()
        sorted_result == sorted_keywords
      end)
      |> Enum.reduce({0, 0}, fn correct?, {correct_count, total_count} ->
        if correct? do
          {correct_count + 1, total_count + 1}
        else
          {correct_count, total_count + 1}
        end
      end)

    percentage = correct_count / total_count * 100
    %{correct: correct_count, total: total_count, percentage: percentage}
  end

  @doc false
  def eval(examples, params, model_info, tokenizer) do
    [{tokens, embeddings, _labels}] = prepare_data(examples, model_info, tokenizer)

    params
    |> forward(embeddings)
    |> Nx.to_flat_list()
    |> Enum.zip(tokens)
    |> Enum.filter(fn {prob, _token} -> prob > 0.5 end)
    |> Enum.map(fn {_prob, token} -> token end)
    |> convert_tokens_to_string()
  end

  @doc false
  def convert_tokens_to_string(tokens) do
    tokens
    |> Enum.join(" ")
    |> String.replace(" ##", "")
    |> String.trim()
  end
end
