defmodule Example.History do
  use Ecto.Schema

  import Ecto.Query
  import Ecto.Changeset
  import Pgvector.Ecto.Query

  alias __MODULE__

  schema "histories" do
    field(:movies, :string)
    field(:embedding, Pgvector.Ecto.Vector)

    timestamps()
  end

  @required_attrs [:movies, :embedding]

  def changeset(message, params \\ %{}) do
    message
    |> cast(params, @required_attrs)
    |> validate_required(@required_attrs)
  end

  def search(embedding) do
    Example.Repo.all(from h in History, order_by: cosine_distance(h.embedding, ^embedding), limit: 3)
  end
end
