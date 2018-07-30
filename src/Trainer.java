import edu.umass.cs.mallet.grmm.types.Factor;
import edu.umass.cs.mallet.grmm.types.LogTableFactor;
import edu.umass.cs.mallet.grmm.types.Variable;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;

/**
 *  This is a class putting node and edge features into table factors. The feature definition is specified in the class
 *  "FeatureGenerator".
 */

public class Trainer {

    // Create a feature generator.
    FeatureGenerator featureGen = new FeatureGenerator();

    public ArrayList<String4Learning> string4Learning(ArrayList<String> sample_string,
                                                      ArrayList<ArrayList<Integer>> label_vec){

        // Each string is stored as an object specifying features as table factors.
        ArrayList<String4Learning> str_list = new ArrayList<>();

        // This is a set of node feature vectors. Keys of the hash map are the feature type indices. The vectors are of
        // the same length equivalent to the string length. This is for a single sequence.
        HashMap<Integer, ArrayList<Double>> list_node_feature = new HashMap<>();

        int num_sample; // number of strings
        int num_label; // number of labels (number of possible outcomes of variables)
        int num_node_feature_type;  // number of node feature types
        int num_edge_feature_type;  // number of edge feature types
        ArrayList<Integer> len_string = new ArrayList<>(); // length of each string

//        // Convert the labels into vectors.
//        label_vec = label_to_vector(label_path);
//
//        // Read the strings into an array.
//        try(BufferedReader br = new BufferedReader(new FileReader(sample_path))){
//            String line;
//            while ((line = br.readLine()) != null){
//                sample_string.add(line);
//            }
//        }catch(Exception e){
//            System.out.println("The training file doesn't exist.");
//        }

        // Get the size of label dictionary.
        num_label = featureGen.dict_label.size();

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

            // Declare variables.
            // Do we need to set different number of outcomes for different node variables?
            Variable[] allVars = new Variable[len_string.get(idx_sample)];
            for(int i=0; i<len_string.get(idx_sample); i++){
                allVars[i] = new Variable(num_label);
            }

            // Add node features.
            list_node_feature = featureGen.getNodeFeature(sample_string.get(idx_sample));
            for(int i=0; i<len_string.get(idx_sample); i++){
                for(int j=0; j<num_node_feature_type; j++){
                    double[] feature_value_arr = new double[num_label];
                    // Node features and transition(edge) features are indexed separately.
                    ArrayList<Double> feature_vector = list_node_feature.get(j);   // for the j-th feature type
                    for(int k=0; k<num_label; k++){
                        feature_value_arr[k] = feature_vector.get(i);
                    }
                    Factor ptl = LogTableFactor.makeFromValues(new Variable[] {allVars[i]}, feature_value_arr);
                    factorList.add(ptl);
                }
            }

            // Create edge features.
            double feature_value;
            double[] feature_value_arr = double[num_label*num_label];
            for(Integer idx1: featureGen.dict_label.keySet()){
                for(Integer idx2: featureGen.dict_label.keySet()){
                    if(featureGen.B_to_I(featureGen.dict_label.get(idx1),featureGen.dict_label.get(idx2))){
                        feature_value = 1;
                    }else{
                        feature_value = 0;
                    }
                    feature_value_arr.add(feature_value);
                }
            }

            // Add edge features.
            for(int i=0; i<num_edge_feature_type; i++){
                HashMap<Integer[], ArrayList<Double>> feature_map = list_edge_feature.get(i); // for the i-th feature type
                for(Integer[] node_set: feature_map.keySet()){
                    // Get the feature vector and store it in an array.
                    ArrayList<Double> feature_vector = feature_map.get(node_set);
                    double[] feature_value_arr = new double[feature_vector.size()];
                    for(int k=0; k<feature_vector.size(); k++){
                        feature_value_arr[k] = feature_vector.get(k);
                    }
                    // Get the variable group (connected nodes).
                    Variable[] current_group = new Variable[node_set.length];
                    for(int k=0; k<node_set.length; k++){
                        current_group[k] = allVars[node_set[k]];
                    }
                    // Create a table factor.
                    Factor ptl = LogTableFactor.makeFromValues(current_group, feature_value_arr);
                    factorList.add(ptl);
                }
            }

            // Add the list of table factors into the sample object.
            String4Learning str = new String4Learning();
            str_list.add(str);
        }

        return str_list;
    }

    // Read a file of ground-truth labels and convert them to vectors.
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

    // Read a file of string sequences (tokens) and convert them to vectors.
    public ArrayList<ArrayList<Integer>> token_to_vector(String filepath){
        ArrayList<ArrayList<Integer>> token_vector = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
            String line;
            String[] tokens;
            ArrayList<Integer> token_vector_tmp;
            int token_tmp;
            while ((line = br.readLine()) != null) {
                tokens = line.split(",");
                token_vector_tmp = new ArrayList<>();
                for(String s: tokens){
                    token_tmp = featureGen.currentToken(s);
                    token_vector_tmp.add(token_tmp);
                }
                token_vector.add(token_vector_tmp);
            }
        } catch (Exception e){
            System.out.println("File doesn't exist.");
        }
        return token_vector;
    }

    public static void main (){
        // To-do: main function for training.
    }

}
