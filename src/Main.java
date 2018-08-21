import edu.umass.cs.mallet.base.classify.evaluate.Graph;
import edu.umass.cs.mallet.grmm.inference.Inferencer;
import edu.umass.cs.mallet.grmm.inference.TRP;
import edu.umass.cs.mallet.grmm.types.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Vector;

public class Main {

    public static void main(String args[]){

        Trainer m_trainer;
        GraphLearner m_graphLearner;

        ArrayList<String> training_string = new ArrayList<>();
        ArrayList<String> testing_string = new ArrayList<>();
        ArrayList<ArrayList<Integer>> training_label = new ArrayList<>();
        ArrayList<ArrayList<String>> testing_label = new ArrayList<>();

        ArrayList<String4Learning> training_data;
        ArrayList<String4Learning> testing_data;

        ArrayList<FactorGraph> testGraphSet;
        ArrayList<ArrayList<Integer>> testPrediction;

        // Read training strings.
        try (BufferedReader br = new BufferedReader(new FileReader("data/train_string.txt"))) {
            String line;
            while ((line = br.readLine()) != null) {
                training_string.add(line);
            }
        } catch (Exception e){
            System.out.println("File doesn't exist.");
        }

        // Read training labels through the trainer.
        m_trainer = new Trainer();
        training_label = m_trainer.label_to_vector("data/train_label.txt");
        //System.out.println(training_label.get(0));
        //System.out.println(m_trainer.featureGen.dict_label.size());

        // Read testing strings.
        try (BufferedReader br = new BufferedReader(new FileReader("data/test_string.txt"))) {
            String line;
            while ((line = br.readLine()) != null) {
                testing_string.add(line);
            }
        } catch (Exception e){
            System.out.println("File doesn't exist.");
        }

        // Read testing labels.
        try (BufferedReader br = new BufferedReader(new FileReader("data/test_label.txt"))) {
            String line;
            String[] labels;
            ArrayList<String> label_seq;
            while ((line = br.readLine()) != null) {
                labels = line.split(",");
                label_seq = new ArrayList<>();
                for(String s: labels){
                    label_seq.add(s);
                }
                testing_label.add(label_seq);
            }
        } catch (Exception e){
            System.out.println("File doesn't exist.");
        }

        // Create customized training data (String4Learning).
        training_data = m_trainer.string4Learning(training_string,training_label);

        // Build up a graph learner and train it using training data.
        m_graphLearner = new GraphLearner(training_data);
        m_graphLearner.doTraining(1);
        for(int i=0; i<10; i++)System.out.println(m_graphLearner.getParameter(i));  // Print some weights.

        // Apply the trained model to the test set.
        testing_data = m_trainer.string4Learning(testing_string,null);
        testGraphSet = m_graphLearner.buildFactorGraphs_test(testing_data);
        testPrediction = m_graphLearner.doTesting(testGraphSet);
        for(int i=0; i<14; i++)System.out.println(testPrediction.get(0).get(i));

    }

}
