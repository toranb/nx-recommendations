defmodule Example.Classifier do
  import Nx.Defn

  def create_vocabulary(examples) do
    examples
    |> Enum.flat_map(fn {desc, _} -> String.split(desc) end)
    |> Enum.concat(Enum.map(examples, fn {_, movie} -> movie end))
    |> Enum.uniq()
    |> Enum.with_index()
    |> Map.new(fn {token, index} -> {token, index} end)
  end

  def preprocess_examples(examples, vocabulary) do
    examples
    |> Enum.map(fn {desc, movie} ->
      desc_ids = desc |> String.split() |> Enum.map(&Map.get(vocabulary, &1, 0))
      movie_id = Map.get(vocabulary, movie, 0)
      {desc_ids, movie_id}
    end)
  end

  defn initialize_embeddings(opts \\ []) do
    dims = opts[:dims]
    vocab_size = opts[:vocab_size]
    key = Nx.Random.key(42)
    Nx.Random.normal_split(key, 0.0, 0.1, shape: {vocab_size, dims}, type: :f16)
  end

  defn initialize_dense(opts \\ []) do
    input_dim = opts[:input_dim]
    output_dim = opts[:output_dim]
    key = Nx.Random.key(42)
    w = Nx.Random.normal_split(key, 0.0, 0.1, shape: {input_dim, output_dim}, type: :f16)
    b = Nx.broadcast(0.0, {output_dim})
    {w, b}
  end

  defn softmax(logits) do
    Nx.exp(logits) / Nx.sum(Nx.exp(logits))
  end

  defn forward({embeddings, w, b}, token_ids) do
    token_ids
    |> embed_and_mean(embeddings)
    |> Nx.dot(w)
    |> Nx.add(b)
    |> softmax()
  end

  defn loss(params, token_ids, movie_id) do
    probs = forward(params, token_ids)
    num_classes = Nx.axis_size(probs, 0)
    one_hot = Nx.equal(Nx.iota({num_classes}), movie_id)

    -Nx.sum(Nx.log(probs + 1.0e-10) * one_hot)
  end

  defn update({embeddings, w, b} = params, desc_ids, movie_id, learning_rate) do
    {loss_value, {grad_embeddings, grad_w, grad_b}} =
      value_and_grad(params, &loss(&1, desc_ids, movie_id))

    updated_embeddings = embeddings - grad_embeddings * learning_rate
    updated_w = w - grad_w * learning_rate
    updated_b = b - grad_b * learning_rate

    {{updated_embeddings, updated_w, updated_b}, loss_value}
  end

  defn embed_and_mean(token_ids, embeddings) do
    Nx.take(embeddings, token_ids) |> Nx.mean(axes: [0])
  end

  defn embed_tokens(embeddings, token_ids) do
    Nx.take(embeddings, token_ids)
  end

  def train(examples, num_epochs, learning_rate) do
    dims = 175
    vocabulary = create_vocabulary(examples)
    preprocessed_examples = preprocess_examples(examples, vocabulary)
    vocab_size = map_size(vocabulary)

    embeddings = initialize_embeddings(vocab_size: vocab_size, dims: dims)

    {w, b} = initialize_dense(input_dim: dims, output_dim: vocab_size)
    initial_params = {embeddings, w, b}

    {final_params, _} =
      Enum.reduce(1..num_epochs, {initial_params, 0.0}, fn epoch, {params, _} ->
        {updated_params, total_loss} =
          Enum.reduce(preprocessed_examples, {params, 0.0}, fn {desc_ids, movie_id},
                                                               {current_params, acc} ->
            desc_tensor = Nx.tensor(desc_ids)
            movie_tensor = Nx.tensor(movie_id)

            {new_params, loss} =
              update(current_params, desc_tensor, movie_tensor, learning_rate)

            {new_params, acc + Nx.to_number(loss)}
          end)

        avg_loss = total_loss / length(preprocessed_examples)
        IO.puts("Epoch #{epoch}, Average Loss: #{avg_loss}")
        {updated_params, avg_loss}
      end)

    {final_params, vocabulary}
  end

  def predict({embeddings, w, b}, description, vocabulary) do
    desc_ids =
      description
      |> String.split()
      |> Enum.map(&Map.get(vocabulary, &1, 0))
      |> Nx.tensor()

    logits = forward({embeddings, w, b}, desc_ids)
    predicted_id = Nx.argmax(logits) |> Nx.to_number()

    vocabulary
    |> Enum.find(fn {_, id} -> id == predicted_id end)
    |> elem(0)
  end
end
