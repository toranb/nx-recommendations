defmodule Example.Prediction do
  NimbleCSV.define(DataParser, separator: ",", escape: "\\n")

  def get_pairs() do
    "descriptions.csv"
    |> File.stream!()
    |> DataParser.parse_stream()
    |> Stream.map(fn [title, desc] ->
      {desc, title}
    end)
    |> Enum.to_list()
  end

  def train() do
    examples = get_pairs()

    num_epochs = 75
    learning_rate = 0.05

    {model_params, _vocabulary} = Example.Classifier.train(examples, num_epochs, learning_rate)
    serialized_container = Nx.serialize(model_params)
    File.write!("#{Path.dirname(__ENV__.file)}/classifier_data", serialized_container)
  end

  def predict_title(description) do
    examples = get_pairs()
    vocabulary = Example.Classifier.create_vocabulary(examples)

    model_data = File.read!("#{Path.dirname(__ENV__.file)}/classifier_data")
    model_params = Nx.deserialize(model_data)
    Example.Classifier.predict(model_params, description, vocabulary)
  end

  def predict_and_recommend(description) do
    description
    |> predict_title()
    |> Example.Recommendation.guess()
  end
end
