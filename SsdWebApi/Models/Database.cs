using System;
using Microsoft.Data.Sqlite;

namespace SsdWebApi.Repositories
{
  public class Persistence
  {
    // public static void Main() {
    //   new Persistence();
    // }

    public Persistence()
    {
      testDB();
    }

    private void testDB()
    {
      var connectionStringBuilder = new SqliteConnectionStringBuilder();
      //Use DB in project directory.  If it does not exist, create it:
      connectionStringBuilder.DataSource = "./testDB.sqlite";
      using (var connection = new SqliteConnection(connectionStringBuilder.ConnectionString))
      {
        connection.Open();

        //Create a table (drop if already exists first):
        var delTableCmd = connection.CreateCommand(); delTableCmd.CommandText = "DROP TABLE IF EXISTS cronistoria";
        delTableCmd.ExecuteNonQuery();
        var createTableCmd = connection.CreateCommand();
        createTableCmd.CommandText = "CREATE TABLE cronistoria(id INTEGER PRIMARY KEY, anno int, serie text)";
        createTableCmd.ExecuteNonQuery();

        //Seed some data:
        using (var transaction = connection.BeginTransaction())
        {
          var insertCmd = connection.CreateCommand();
          insertCmd.CommandText = "INSERT INTO cronistoria (anno,serie) VALUES(2014,'A')";
          insertCmd.ExecuteNonQuery();
          insertCmd.CommandText = "INSERT INTO cronistoria (anno,serie) VALUES(2015,'B')";
          insertCmd.ExecuteNonQuery();
          insertCmd.CommandText = "INSERT INTO cronistoria (anno,serie) VALUES(2016,'B')";
          insertCmd.ExecuteNonQuery();
          insertCmd.CommandText = "INSERT INTO cronistoria (anno,serie) VALUES(2017,'B')";
          insertCmd.ExecuteNonQuery();
          insertCmd.CommandText = "INSERT INTO cronistoria (anno,serie) VALUES(2018,'D')";
          insertCmd.ExecuteNonQuery();
          transaction.Commit();
        }

        //Read the newly inserted data:
        var selectCmd = connection.CreateCommand();
        selectCmd.CommandText = "SELECT anno, serie FROM cronistoria";
        using (var reader = selectCmd.ExecuteReader())
        {
          while (reader.Read())
          {
            int anno = Convert.ToInt32(reader.GetString(0));
            var message = anno + " " + (anno + 1) + " Serie " + reader.GetString(1);
            Console.WriteLine(message);
          }
        }
      }
    }
  }
}
