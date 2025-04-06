defmodule Example.Translate do
  import Nx.Defn

  @moduledoc false

  @max_length 32
  @end_token_id 102
  @start_token_id 101
  @vocab_size 30_522
  @model "google-bert/bert-base-uncased"

  NimbleCSV.define(FluentParser, separator: ",", escape: "\"")

  def go() do
    num_epochs = 19
    learning_rate = 0.04
    model = train(num_epochs, learning_rate)

    model_params = %{model: model}
    serialized_container = Nx.serialize(model_params)
    File.write!("#{Path.dirname(__ENV__.file)}/slang_weights", serialized_container)
  end

  def test(text, opts \\ []) do
    model_data = File.read!("#{Path.dirname(__ENV__.file)}/slang_weights")
    deserialized_data = Nx.deserialize(model_data)
    predict(deserialized_data.model, text, opts)
  end

  def training_data() do
    "slang.csv"
    |> File.stream!()
    |> FluentParser.parse_stream(skip_headers: false)
    |> Stream.map(fn [source, target] ->
      {source, target}
    end)
    |> Enum.to_list()
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

    serving =
      Example.Generation.get_text(model_info, tokenizer, generation_config,
        compile: [batch_size: 512, sequence_length: [@max_length]],
        defn_options: [compiler: EXLA]
      )

    source_texts = examples |> Enum.map(fn {source, _} -> source end)
    %{hidden_states: source_hidden_state} = Nx.Serving.run(serving, source_texts)

    target_texts = examples |> Enum.map(fn {_, target} -> target end)
    %{tokens: target_tokens} = Nx.Serving.run(serving, target_texts)

    0..(elem(Nx.shape(source_hidden_state), 0) - 1)
    |> Enum.map(fn i ->
      target_ids = Nx.slice(target_tokens, [i, 0], [1, @max_length]) |> Nx.to_flat_list()

      source_embs =
        Nx.slice(source_hidden_state, [i, 0, 0], [1, @max_length, 768]) |> Nx.squeeze(axes: [0])

      {source_embs, Nx.tensor(target_ids)}
    end)
  end

  defn one_hot(indices, num_classes) do
    indices
    |> Nx.reshape({:auto, 1})
    |> Nx.equal(Nx.iota({num_classes}))
  end

  defn softmax(logits, opts \\ []) do
    axes = opts[:axes]
    e_x = Nx.exp(logits - Nx.reduce_max(logits, axes: axes, keep_axes: true))
    e_x / Nx.sum(e_x, axes: axes, keep_axes: true)
  end

  defn relu(x) do
    custom_grad(
      Nx.max(x, 0),
      [x],
      fn g -> [Nx.select(Nx.greater(x, 0), g, Nx.broadcast(0, g))] end
    )
  end

  defn initialize_seq2seq(opts \\ []) do
    input_dim = opts[:input_dim]
    hidden_dim = opts[:hidden_dim]
    vocab_size = opts[:vocab_size]

    key_one = Nx.Random.key(42)
    key_two = Nx.Random.key(23)
    key_three = Nx.Random.key(17)
    key_four = Nx.Random.key(11)
    key_five = Nx.Random.key(7)

    enc_w1 = Nx.Random.normal_split(key_one, 0.0, 0.1, shape: {input_dim, hidden_dim}, type: :f32)
    enc_b1 = Nx.broadcast(0.0, {hidden_dim})

    dec_w1 =
      Nx.Random.normal_split(key_two, 0.0, 0.1, shape: {hidden_dim, hidden_dim}, type: :f32)

    dec_b1 = Nx.broadcast(0.0, {hidden_dim})

    attn_w =
      Nx.Random.normal_split(key_three, 0.0, 0.1, shape: {hidden_dim, hidden_dim}, type: :f32)

    attn_b = Nx.broadcast(0.0, {hidden_dim})

    out_w =
      Nx.Random.normal_split(key_four, 0.0, 0.1, shape: {hidden_dim, vocab_size}, type: :f32)

    out_b = Nx.broadcast(0.0, {vocab_size})

    emb_w =
      Nx.Random.normal_split(key_five, 0.0, 0.1, shape: {vocab_size, hidden_dim}, type: :f32)

    {enc_w1, enc_b1, dec_w1, dec_b1, attn_w, attn_b, out_w, out_b, emb_w}
  end

  defn forward(params, source_embeddings, target_ids) do
    {target_len} = Nx.shape(target_ids)
    {enc_w1, enc_b1, dec_w1, dec_b1, attn_w, attn_b, out_w, out_b, emb_w} = params

    encoded_source =
      source_embeddings
      |> Nx.dot(enc_w1)
      |> Nx.add(enc_b1)
      |> relu()

    decoder_input_ids = Nx.slice(target_ids, [0], [target_len - 1])
    decoder_embed = Nx.dot(one_hot(decoder_input_ids, @vocab_size), emb_w)
    queries = Nx.dot(decoder_embed, attn_w) + attn_b

    keys = encoded_source
    values = encoded_source
    attn_scores = Nx.dot(queries, Nx.transpose(keys, axes: [1, 0]))

    attn_weights = softmax(attn_scores, axes: [-1])
    context = Nx.dot(attn_weights, values)

    combined = context + decoder_embed
    decoded = (Nx.dot(combined, dec_w1) + dec_b1) |> relu()

    logits = Nx.dot(decoded, out_w) + out_b
    softmax(logits, axes: [-1])
  end

  defn loss(params, source_embeddings, target_ids) do
    {target_len} = Nx.shape(target_ids)

    token_probs = forward(params, source_embeddings, target_ids)
    actual_target_ids = Nx.slice(target_ids, [1], [target_len - 1])
    target_one_hot = one_hot(actual_target_ids, @vocab_size)

    log_probs = Nx.log(token_probs + 1.0e-9)
    position_losses = -Nx.sum(target_one_hot * log_probs, axes: [-1])

    Nx.mean(position_losses)
  end

  defn update(params, source_embeddings, target_ids, learning_rate) do
    {loss_value, gradients} =
      value_and_grad(params, &loss(&1, source_embeddings, target_ids))

    {enc_w1, enc_b1, dec_w1, dec_b1, attn_w, attn_b, out_w, out_b, emb_w} = params

    {grad_enc_w1, grad_enc_b1, grad_dec_w1, grad_dec_b1, grad_attn_w, grad_attn_b, grad_out_w,
     grad_out_b, grad_emb_w} = gradients

    updated_params = {
      enc_w1 - grad_enc_w1 * learning_rate,
      enc_b1 - grad_enc_b1 * learning_rate,
      dec_w1 - grad_dec_w1 * learning_rate,
      dec_b1 - grad_dec_b1 * learning_rate,
      attn_w - grad_attn_w * learning_rate,
      attn_b - grad_attn_b * learning_rate,
      out_w - grad_out_w * learning_rate,
      out_b - grad_out_b * learning_rate,
      emb_w - grad_emb_w * learning_rate
    }

    {updated_params, loss_value}
  end

  defn decode_step(params, source_embeddings, last_token_tensor) do
    {enc_w1, enc_b1, dec_w1, dec_b1, attn_w, attn_b, out_w, out_b, emb_w} = params

    encoded_source =
      source_embeddings
      |> Nx.dot(enc_w1)
      |> Nx.add(enc_b1)
      |> relu()

    last_token_one_hot = one_hot(last_token_tensor, @vocab_size)
    decoder_input_embedding = Nx.dot(last_token_one_hot, emb_w)

    query = Nx.dot(decoder_input_embedding, attn_w) + attn_b

    keys = encoded_source
    values = encoded_source
    attn_scores = Nx.dot(query, Nx.transpose(keys, axes: [1, 0]))

    attn_weights = softmax(attn_scores, axes: [-1])
    context = Nx.dot(attn_weights, values)

    combined = context + decoder_input_embedding

    decoded =
      combined
      |> Nx.dot(dec_w1)
      |> Nx.add(dec_b1)
      |> relu()

    decoded
    |> Nx.dot(out_w)
    |> Nx.add(out_b)
  end

  def train(num_epochs, learning_rate) do
    examples = training_data()

    {:ok, model_info} = Bumblebee.load_model({:hf, @model}, architecture: :base)
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model})

    bert_dim = 768
    hidden_dim = 128

    preprocessed_examples = prepare_data(examples, model_info, tokenizer)

    initial_params =
      initialize_seq2seq(input_dim: bert_dim, hidden_dim: hidden_dim, vocab_size: @vocab_size)

    Enum.reduce(1..num_epochs, initial_params, fn epoch, epoch_params ->
      example_len = floor(length(preprocessed_examples) * 0.9)
      {training_data, validation_data} = Enum.split(preprocessed_examples, example_len)

      shuffled_examples = Enum.shuffle(training_data)
      batches = Enum.chunk_every(shuffled_examples, 16)

      {final_epoch_params, total_loss} =
        Enum.reduce(batches, {epoch_params, 0.0}, fn batch, {current_params, acc_loss} ->
          {batch_params, batch_loss} =
            Enum.reduce(batch, {current_params, 0.0}, fn {source_embs, target_ids},
                                                         {params, loss} ->
              {new_params, example_loss} = update(params, source_embs, target_ids, learning_rate)

              # Important: Only deallocate the intermediate params, not the ones we'll return
              if params != current_params do
                Nx.backend_deallocate(params)
              end

              {new_params, loss + Nx.to_number(example_loss)}
            end)

          # Important: Only deallocate the intermediate params, not the ones we'll return
          if current_params != epoch_params do
            Nx.backend_deallocate(current_params)
          end

          avg_batch_loss = batch_loss / length(batch)
          {batch_params, acc_loss + avg_batch_loss}
        end)

      if rem(epoch, 2) == 0 do
        shuffled_validation_examples = Enum.shuffle(validation_data)
        validate_batch = Enum.take(shuffled_validation_examples, 16)

        validation_loss =
          Enum.reduce(validate_batch, 0.0, fn {source_embs, target_ids}, val_acc_loss ->
            val_loss = loss(final_epoch_params, source_embs, target_ids)
            val_acc_loss + Nx.to_number(val_loss)
          end)

        avg_epoch_loss = total_loss / length(batches)
        avg_validation_loss = validation_loss / length(validate_batch)

        IO.puts(
          "Epoch #{epoch}, Train Loss: #{avg_epoch_loss}, Validation Loss: #{avg_validation_loss}"
        )
      else
        avg_epoch_loss = total_loss / length(batches)
        IO.puts("Epoch #{epoch}, Avg Loss: #{avg_epoch_loss}")
      end

      final_epoch_params
    end)
  end

  defn downweight_previous_token(logits, prev_token_id, penalty \\ 0.9) do
    mask = Nx.iota(Nx.shape(logits)) == prev_token_id
    Nx.select(mask, Nx.multiply(logits, penalty), logits)
  end

  def predict(params, text, opts \\ []) do
    max_tokens = Keyword.get(opts, :max_length, @max_length)
    temperature = Keyword.get(opts, :temperature, 0.9)

    {:ok, model_info} = Bumblebee.load_model({:hf, @model}, architecture: :base)
    {:ok, bert_tokenizer} = Bumblebee.load_tokenizer({:hf, @model})

    tokenizer =
      Bumblebee.configure(bert_tokenizer,
        pad_direction: :right,
        return_token_type_ids: false,
        return_length: false
      )

    {:ok, config} = Bumblebee.load_generation_config({:hf, @model})
    generation_config = Bumblebee.configure(config, max_new_tokens: @max_length)

    serving =
      Example.Generation.get_text(model_info, tokenizer, generation_config,
        compile: [batch_size: 512, sequence_length: [@max_length]],
        defn_options: [compiler: EXLA]
      )

    %{hidden_states: source_hidden_state} = Nx.Serving.run(serving, [text])

    source_embeddings =
      Nx.slice(source_hidden_state, [0, 0, 0], [1, @max_length, 768]) |> Nx.squeeze(axes: [0])

    initial_ids = [@start_token_id]

    gen_token_ids =
      Stream.iterate({initial_ids, initial_ids, 0}, fn {token_ids, next_token, count} ->
        logits = decode_step(params, source_embeddings, Nx.tensor(next_token))
        scaled_logits = Nx.divide(logits, temperature)

        scaled_logits =
          if count > 0 do
            prev_token_id = List.last(token_ids)
            downweight_previous_token(scaled_logits, prev_token_id)
          else
            scaled_logits
          end

        probs = softmax(scaled_logits, axes: [-1])
        next_token = probs |> Nx.argmax() |> Nx.to_number()

        {token_ids ++ [next_token], [next_token], count + 1}
      end)
      |> Enum.reduce_while(initial_ids, fn {token_ids, _, count}, _acc ->
        last_token = List.last(token_ids)

        if last_token == @end_token_id or count >= max_tokens do
          {:halt, token_ids}
        else
          {:cont, token_ids}
        end
      end)

    Bumblebee.Tokenizer.decode(tokenizer.native_tokenizer, gen_token_ids)
  end
end
