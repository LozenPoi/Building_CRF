import edu.umass.cs.mallet.base.classify.evaluate.Graph;
import edu.umass.cs.mallet.grmm.inference.Inferencer;
import edu.umass.cs.mallet.grmm.inference.TRP;
import edu.umass.cs.mallet.grmm.types.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Vector;

public class Main {

//    public double doTesting(ArrayList<FactorGraph> testGraphSet, ArrayList<Assignment>testAssignment){
//        FactorGraph graph;
//        Assignment assin, guess;
//        AssignmentIterator it;
//        Factor ptl;
//        Variable variable;
//        int varSize, var, parID = 0;
//        int[] prediction;
//        double max, correct = 0, total = 0, TP = 0, LL = 0;
//        Vector<Double> acc = new Vector<>();
//        String filename;
//
//        Inferencer map_infer = TRP.createForMaxProduct();
//        //Inferencer map_infer = LoopyBP.createForMaxProduct();
//        for(int sampleID=0; sampleID<m_trainSampleSet.size(); sampleID++) {
//            graph = testGraphSet.get(sampleID);
//            assin = testAssignment.get(sampleID);
//            varSize = graph.numVariables();
//            correct = 0;
//            prediction = new int[varSize];
//            map_infer.computeMarginals(graph);  //begin to collect the expectations
//            for(var=1; var<varSize; var++) {
//                //retrieve the MAP configuration
//                variable = graph.get(var);
//                ptl = map_infer.lookupMarginal( variable );
//                max = -Double.MAX_VALUE;
//                for (it = ptl.assignmentIterator (); it.hasNext (); it.next()) {
//                    if (ptl.value(it)>max) {
//                        max = ptl.value(it);
//                        parID = it.indexOfCurrentAssn();
//                    }
//                }
//                prediction[var] = parID;
//                //evaluate the performance
//                if( parID == assin.get(variable) )
//                    correct++;
//            }
//
//            guess = new Assignment(graph, prediction);
//            if ( map_infer.lookupLogJoint(guess) > map_infer.lookupLogJoint(assin) )
//                LL++;
//
//            prediction[0] = -1;
//            if (resultpath!=null){
//                filename = m_trainSampleSet.get(sampleID).threadid.get(0);
//                outPrediction(resultpath + filename + ".res", prediction);
//            }
//
//            acc.add(correct/(varSize-1));
//            TP += correct;
//            total += varSize-1;
//        }
//
//        correct = 0;
//        for(parID=0; parID<acc.size(); parID++)
//            correct += acc.get(parID).doubleValue();
//        System.out.println("Micro accuracy " + TP/total);
//        System.out.println("Macro accuracy " + correct/acc.size());
//        System.out.println(LL/acc.size() + " percentage threads have better configuration in likelihood!");
//        return TP/total;
//    }

    public static void main(String args[]){

        Trainer m_trainer;
        GraphLearner m_graphLearner;

        ArrayList<String> training_string = new ArrayList<>();
        ArrayList<String> testing_string = new ArrayList<>();
        ArrayList<ArrayList<Integer>> training_label = new ArrayList<>();
        ArrayList<ArrayList<String>> testing_label = new ArrayList<>();

        ArrayList<String4Learning> training_data = new ArrayList<>();

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

        // Build up a graph learner.
        m_graphLearner = new GraphLearner(training_data);
        m_graphLearner.doTraining(10);
        for(int i=0; i<10; i++)System.out.println(m_graphLearner.getParameter(i));

    }

}
