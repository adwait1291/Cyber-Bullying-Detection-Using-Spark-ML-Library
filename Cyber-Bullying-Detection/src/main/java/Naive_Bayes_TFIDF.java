import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.*;

public class Naive_Bayes_TFIDF {
	public static void main(String[] args) {
		SparkSession spark = SparkSession
			      .builder()
			      .appName("Naive_Bayes")
			      .master("local[*]")
			      .getOrCreate();

		//---------------------------Loading Dataset---------------------//
		String path = "data/data.csv";
		Dataset<Row> df = spark.read().option("header", "true").csv(path);
		df.show(false);

        	df = df.select(
        		df.col("comments"),
        		df.col("tagging").cast(DataTypes.IntegerType)
        	);		
	    
	    	Dataset<Row>describe = df.describe();
	    	describe.show();

	    	Tokenizer tokenizer = new Tokenizer().setInputCol("comments").setOutputCol("words");
	    	df = tokenizer.transform(df);
	    
	    
	  	//---------------------------Splitting into train and test set---------------------//
	    	Dataset<Row>[] BothTrainTest = df.randomSplit(new double[] {0.8d,0.2d},42);
		Dataset<Row> TrainDf = BothTrainTest[0];
		Dataset<Row> TestDf = BothTrainTest[1];


		//---------------------------Count Vectorizer---------------------//
		CountVectorizerModel cvModel = new CountVectorizer()
				.setInputCol("words")
				.setOutputCol("Count Vectorizer")
				.setVocabSize(20000)
				.setMinDF(2)
				.fit(TrainDf);
		
	    	TrainDf = cvModel.transform(TrainDf);
	    	TestDf = cvModel.transform(TestDf);
	    
	    	IDF idf = new IDF().setInputCol("Count Vectorizer").setOutputCol("TF-IDF Vectorizer");
	    	IDFModel idfModel = idf.fit(TrainDf);
	    
	    	TrainDf = idfModel.transform(TrainDf);	  
	    	TestDf = idfModel.transform(TestDf);
		TrainDf.show(false);
	    

	    	VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"TF-IDF Vectorizer"})
				.setOutputCol("features");
	    	TrainDf = assembler.transform(TrainDf);
	    	TestDf = assembler.transform(TestDf);	
	    
       
	  	//---------------------------Model Training---------------------//
	    	NaiveBayes nb = new NaiveBayes().setLabelCol("tagging");
	    	NaiveBayesModel model = nb.fit(TrainDf);
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
