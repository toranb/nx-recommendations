defmodule Example.Embedding do
  import Nx.Defn

  def create_vocabulary(examples) do
    examples
    |> Enum.flat_map(&String.split/1)
    |> Enum.uniq()
    |> Enum.with_index()
    |> Map.new(fn {token, index} -> {token, index} end)
  end

  def preprocess_examples(examples, vocabulary) do
    examples
    |> Enum.flat_map(fn example ->
      movies = String.split(example)
      token_ids = Enum.map(movies, &Map.get(vocabulary, &1, 0))
      for i <- 0..(length(movies) - 1),
          j <- 0..(length(movies) - 1),
          i != j,
          do: {Enum.at(token_ids, i), Enum.at(token_ids, j)}
    end)
    |> Enum.uniq()
  end

  def tokenize(text, vocabulary) do
    text
    |> String.split()
    |> Enum.map(&Map.get(vocabulary, &1, 0))
  end

  deftransform to_float_type(tensor) do
    tensor |> Nx.type() |> Nx.Type.to_floating()
  end

  defn normalize(tensor) do
    cutoff = 10 * Nx.Constants.epsilon(:f64)

    norm =
      tensor
      |> Nx.pow(2)
      |> Nx.sum()
      |> Nx.sqrt()

    selected_norm = Nx.select(norm > cutoff, norm, Nx.tensor(1.0, type: to_float_type(tensor)))

    Nx.divide(tensor, selected_norm)
  end

  defn cosine_similarity(a, b) do
    distance = Scholar.Metrics.Distance.cosine(a, b)
    Nx.subtract(Nx.tensor(1), distance)
  end

  defn pair_loss(embeddings, id_one, id_two) do
    one = embeddings |> embed_tokens(id_one)
    two = embeddings |> embed_tokens(id_two)
    similarity = cosine_similarity(one, two)
    -similarity
  end

  defn update(embeddings, id_one, id_two, learning_rate) do
    {loss_value, gradients} = value_and_grad(embeddings, fn emb ->
      pair_loss(emb, id_one, id_two)
    end)

    updated_embeddings = Nx.subtract(embeddings, Nx.multiply(learning_rate, gradients))
    {updated_embeddings, loss_value}
  end

  defn initialize_embeddings(opts \\ []) do
    dims = opts[:dims]
    vocab_size = opts[:vocab_size]
    key = Nx.Random.key(42)
    Nx.Random.normal_split(key, 0.0, 0.1, shape: {vocab_size, dims}, type: :f16)
  end

  defn embed_tokens(embeddings, token_ids) do
    Nx.take(embeddings, token_ids)
  end

  def train(data, num_epochs, learning_rate) do
    dims = 25
    vocabulary = create_vocabulary(data)
    examples = preprocess_examples(data, vocabulary)
    vocab_size = length(examples)
    embeddings = initialize_embeddings(vocab_size: vocab_size, dims: dims)

    Enum.reduce(1..num_epochs, embeddings, fn epoch, emb ->
      {updated_emb, total_loss} = Enum.reduce(examples, {emb, 0}, fn {one, two}, {current_emb, acc} ->
          {new_emb, loss} = update(current_emb, one, two, learning_rate)
          {new_emb, acc + Nx.to_number(loss)}
        end)

      result = "Epoch #{epoch}, Examples: #{length(examples)}, Loss: #{total_loss / length(examples)}"
      IO.inspect(result)

      updated_emb
    end)
  end

  def get_embedding(movies, vocabulary, embeddings) do
    token_ids = movies
                |> String.split()
                |> Enum.map(&Map.get(vocabulary, &1, 0))
                |> Nx.tensor()

    embeddings
    |> embed_tokens(token_ids)
    |> Nx.mean(axes: [0])
    |> Nx.to_flat_list()
  end
end
