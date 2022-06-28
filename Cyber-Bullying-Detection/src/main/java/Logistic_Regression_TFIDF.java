import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

public class Logistic_Regression_TFIDF {
	public static void main(String[] args) {
		SparkSession spark = SparkSession
			      .builder()
			      .appName("Naive_Bayes")
			      .master("local[*]")
			      .getOrCreate();
		
		//---------------------------Loading Dataset---------------------//
		String path = "data/data.csv";
		Dataset<Row> df = spark.read().option("header", "true").csv(path);
		df.show();
       		df = df.select(
        		df.col("comments"),
        		df.col("tagging").cast(DataTypes.IntegerType)
        	);		
	    
	    	Dataset<Row>describe = df.describe();
	    	describe.show();
	    
	    
	    	Tokenizer tokenizer = new Tokenizer().setInputCol("comments").setOutputCol("words");
	    	df = tokenizer.transform(df);
	    
	    
	  	//---------------------------Splitting into train and test set---------------------//
	   	Dataset<Row>[] BothTrainTest = df.randomSplit(new double[] {0.8d,0.2d});
		Dataset<Row> TrainDf = BothTrainTest[0];
		Dataset<Row> TestDf = BothTrainTest[1];
    
	        
	    	//---------------------------TF-IDF---------------------//
	    	HashingTF hashingTF = new HashingTF()
			      .setInputCol("words")
			      .setOutputCol("rawFeatures")
			      .setNumFeatures(20000);
				
		TrainDf = hashingTF.transform(TrainDf);	  
		TestDf = hashingTF.transform(TestDf);	
			    
		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("feature");
		IDFModel idfModel = idf.fit(TrainDf);
			    
		TrainDf = idfModel.transform(TrainDf);	  
		TestDf = idfModel.transform(TestDf);
		TrainDf.show();    
	    
	    	VectorAssembler assembler = new VectorAssembler()
	    	      .setInputCols(new String[]{"feature"})
	    	      .setOutputCol("features");
	    	TrainDf = assembler.transform(TrainDf);
	    	TestDf = assembler.transform(TestDf);	
	    
       
	  	//---------------------------Model Training---------------------//
	    	LogisticRegression lr = new LogisticRegression()
	    		  .setMaxIter(10)
	    		  .setRegParam(0.3)
	    		  .setElasticNetParam(0.8)
	    		  .setLabelCol("tagging");
	    	LogisticRegressionModel model = lr.fit(TrainDf);
	    	Dataset<Row> predictions = model.transform(TestDf);
	   
	    
	  	//---------------------------Printing Accuracy---------------------//
	    	MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
	    	      .setLabelCol("tagging")
	    	      .setPredictionCol("prediction")
	    	      .setMetricName("accuracy");
	    	double accuracy = evaluator.evaluate(predictions);
	    	System.out.println("Test set accuracy = " + accuracy);
	    
	    
		spark.stop();
	}

}
