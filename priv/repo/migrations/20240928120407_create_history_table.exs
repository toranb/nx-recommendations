defmodule Example.Repo.Migrations.CreateHistoryTable do
  use Ecto.Migration

  def change do
    create table(:histories) do
      add :movies, :text, null: false
      add :embedding, :vector, size: 100

      timestamps()
    end

    create index("histories", ["embedding vector_cosine_ops"], using: :hnsw)
  end
end
