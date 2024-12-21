# Example

Start iex

  * Run `mix setup` to install and setup dependencies
  * Run `iex -S mix run` to get the iex shell
  * Run `Example.Recommendation.train()` to train for 10 epochs
  * Run `Example.Recommendation.guess("Batman")` to see movie recommendations

  * Run `Example.Prediction.train()` to train model to find title from descriptions
  * Run `Example.Prediction.predict_title("fights crime in Gotham")` to predict the movie title
  * Run `Example.Prediction.predict_and_recommend("fights crime in Gotham")` to predict the title and like movies

  * Run `Example.Keywords.go()` to train model to find keywords from ecomm question, keyword pairs
  * Run `Example.Keywords.evaluate()` to evaluate the keyword model
