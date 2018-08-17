import edu.umass.cs.mallet.grmm.types.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 *  This is a class putting node and edge features into table factors. The feature definition is specified in the class
 *  "FeatureGenerator".
 */

public class Trainer {

    // Create a feature generator.
    FeatureGenerator featureGen;

    public Trainer(){
        featureGen = new FeatureGenerator();
    }

    public ArrayList<String4Learning> string4Learning(ArrayList<String> sample_string,
                                                      ArrayList<ArrayList<Integer>> label_vec){

        // Each string is stored as an object specifying features as table factors.
        ArrayList<String4Learning> str_list = new ArrayList<>();

        // This is a set of node feature vectors. Keys of the hash map are the feature type indices. The vectors are of
        // the same length equivalent to the string length. This is for a single sequence.
        HashMap<Integer, ArrayList<Double>> list_node_feature;

        int num_sample; // number of strings
        int num_label; // number of labels (number of possible outcomes of variables)
        int num_node_feature_type;  // number of node feature types
        int num_edge_feature_type;  // number of edge feature types
        ArrayList<Integer> len_string = new ArrayList<>(); // length of each string

        // Get the size of label dictionary.
        num_label = featureGen.dict_label.size();

        // Get the number of node and edge features.
        num_node_feature_type = featureGen.dict_node_feature.size();
        num_edge_feature_type = featureGen.dict_edge_feature.size();

        // Get the length of all the input strings.
        num_sample = sample_string.size();
        for(String s: sample_string){
            len_string.add(s.length());
        }

        // For each training sample, construct a factor graph via variables. and a list of table factors to specify edge
        // and node features.
        for(int idx_sample=0; idx_sample<num_sample; idx_sample++){

            ArrayList<Factor> factorList = new ArrayList<>();   // list of table factors for the current string
            ArrayList<Integer> featureType = new ArrayList<>(); // corresponding feature ID for each list of factors

            // Declare variables.
            // Do we need to set different number of outcomes for different node variables?
            Variable[] allVars = new Variable[len_string.get(idx_sample)];
            for(int i=0; i<len_string.get(idx_sample); i++){
                allVars[i] = new Variable(num_label);
            }

            // Add node features.
            list_node_feature = featureGen.getNodeFeature(sample_string.get(idx_sample));
            ArrayList<Double> feature_vector;
            double[] feature_value_arr = new double[num_label];
            for(int i=0; i<num_node_feature_type; i++){
                feature_vector = list_node_feature.get(i);
                for(int j=0; j<len_string.get(idx_sample); j++){
                    Arrays.fill(feature_value_arr, feature_vector.get(j));
                    Factor ptl = LogTableFactor.makeFromValues(new Variable[] {allVars[j]}, feature_value_arr);
//                    VarSet varSet = new HashVarSet(new Variable[] { allVars[j] });
//                    Factor ptl = LogTableFactor.makeFromValues(varSet, feature_value_arr);
                    factorList.add(ptl);
                    featureType.add(i);
                }
            }

            // Add all first-order transition features f(y_(i-1),y_i).
            double[] trans_feature_arr;
            for(int i=0; i<num_label; i++){
                for(int j=0; j<num_label; j++){
                    trans_feature_arr = featureGen.label_transition(i,j);
                    //System.out.println(trans_feature_arr.toString());
                    for(int k=0; k<len_string.get(idx_sample)-1; k++){
                        Factor ptl = LogTableFactor.makeFromValues(
                                new Variable[] {allVars[k], allVars[k+1]}, trans_feature_arr);
//                        Factor ptl = new TableFactor(
//                                new Variable[] {allVars[k], allVars[k+1]}, trans_feature_arr);
                        factorList.add(ptl);
                        featureType.add(num_node_feature_type+i*num_label+j);
                    }
                }
            }

            // Add the list of table factors into the sample object.
            String4Learning str = new String4Learning(factorList, featureType, label_vec.get(idx_sample));
            str_list.add(str);
        }

        return str_list;
    }

    // Read a file of ground-truth labels and convert them to vectors (this updates the label dictionary).
    public ArrayList<ArrayList<Integer>> label_to_vector(String filepath){
        ArrayList<ArrayList<Integer>> label_vector = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
            String line;
            String[] labels;
            ArrayList<Integer> label_vector_tmp;
            int label_tmp;
            while ((line = br.readLine()) != null) {
                labels = line.split(",");
                label_vector_tmp = new ArrayList<>();
                for(String s: labels){
                    label_tmp = featureGen.update_label(s);
                    label_vector_tmp.add(label_tmp);
                }
                label_vector.add(label_vector_tmp);
            }
        } catch (Exception e){
            System.out.println("File doesn't exist.");
        }
        return label_vector;
    }

//    // Read a file of string sequences (tokens) and convert them to vectors.
//    public ArrayList<ArrayList<Integer>> token_to_vector(String filepath){
//        ArrayList<ArrayList<Integer>> token_vector = new ArrayList<>();
//        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
//            String line;
//            String[] tokens;
//            ArrayList<Integer> token_vector_tmp;
//            int token_tmp;
//            while ((line = br.readLine()) != null) {
//                tokens = line.split(",");
//                token_vector_tmp = new ArrayList<>();
//                for(String s: tokens){
//                    token_tmp = featureGen.currentToken(s);
//                    token_vector_tmp.add(token_tmp);
//                }
//                token_vector.add(token_vector_tmp);
//            }
//        } catch (Exception e){
//            System.out.println("File doesn't exist.");
//        }
//        return token_vector;
//    }

}
