defmodule Example.Recommendation do
  NimbleCSV.define(ZataParser, separator: ",", escape: "\"")

  def get_history() do
    "history.csv"
    |> File.stream!()
    |> ZataParser.parse_stream(skip_headers: false)
    |> Stream.map(fn [movies] ->
      movies
    end)
    |> Enum.to_list()
  end

  def train() do
    examples = get_history()
    vocabulary = Example.Embedding.create_vocabulary(examples)

    num_epochs = 10
    learning_rate = 0.01

    model = Example.Embedding.train(examples, num_epochs, learning_rate)
    serialized_container = Nx.serialize(model)
    File.write!("#{Path.dirname(__ENV__.file)}/model_data", serialized_container)

    # "history.csv"
    "datas.csv"
    |> File.stream!()
    |> ZataParser.parse_stream(skip_headers: false)
    |> Stream.map(fn [movies] ->
      result = Example.Embedding.get_embedding(movies, vocabulary, model) |> Nx.tensor()
      result = result |> Example.Embedding.normalize()

      %Example.History{}
      |> Example.History.changeset(%{
        movies: movies,
        embedding: result
      })
    end)
    |> Enum.each(fn embedding ->
      embedding |> Example.Repo.insert!()
    end)
  end

  def guess(movies) do
    examples = get_history()

    model_data = File.read!("#{Path.dirname(__ENV__.file)}/model_data")
    model = Nx.deserialize(model_data)

    vocabulary = Example.Embedding.create_vocabulary(examples)

    result = Example.Embedding.get_embedding(movies, vocabulary, model) |> Nx.tensor()
    result = result |> Example.Embedding.normalize()
    results = Example.History.search(result)

    results
    |> Enum.each(fn history ->
      vector = history.embedding |> Pgvector.to_tensor()

      norm_one = Example.Embedding.normalize(vector)
      norm_two = Example.Embedding.normalize(result)
      similarity = Nx.dot(norm_one, norm_two)

      similarity |> IO.inspect(label: "similarity")
      history.movies |> IO.inspect(label: "movies")
    end)

    results
  end
end
