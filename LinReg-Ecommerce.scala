////////////////////////////////////////////
//// LINEAR REGRESSION ///////////
/// Building a Model to Predict Ecommerce Customer Spending ///
/////////////////////////////////////////

// Import LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

// Optional: I Use the following code below to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Starting a simple Spark Session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

// Using Spark to read in the Ecommerce Customers csv file.
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Clean_Ecommerce.csv")

// Printing the Schema of the DataFrame
data.printSchema()

// Printing out an example Row

val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example Data Row")
for(ind <- Range(0,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}

////////////////////////////////////////////////////
//// Setting Up DataFrame for Machine Learning ////
//////////////////////////////////////////////////

// A few things we need to do before Spark can accept the data!
// It needs to be in the form of two columns
// ("label","features")

// Importing VectorAssembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Renaming the Yearly Amount Spent Column as "label"
// Also I only grab the numerical columns from the data
// Setting all of this as a new dataframe called df
val df = data.select(data("Yearly Amount Spent").as("label"),
  $"Avg Session Length",$"Time on App",$"Time on Website",$"Length of Membership")

// An assembler converts the input values to a vector
// A vector is what the ML algorithm reads to train a model

// Using VectorAssembler to convert the input columns of df
// to a single output column of an array called "features"
// Setting the input columns from which we are supposed to read the values.
// Calling this new object assembler
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length","Time on App","Time on Website","Length of Membership")).setOutputCol("features")

// Using the assembler to transform our DataFrame to the two columns: label and features
val output = assembler.transform(df).select($"label",$"features")

// Creating a Linear Regression Model object
val lr = new LinearRegression()

// Fitting the model to the data and call this model lrModel
val lrModel = lr.fit(output)

// Printing the coefficients and intercept for linear regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Summarizing the model over the training set and print out some metrics!
// Using the .summary method off my model to create an object called trainingSummary

// Showing the residuals, the RMSE, the MSE, and the R^2 Values.
val trainingSummary = lrModel.summary

println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")

trainingSummary.residuals.show()

println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"r2: ${trainingSummary.r2}")

// The End!
