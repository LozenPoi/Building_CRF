import edu.umass.cs.mallet.grmm.types.Factor;
import edu.umass.cs.mallet.grmm.types.LogTableFactor;
import edu.umass.cs.mallet.grmm.types.Variable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;

/**
 *  This is a class putting node and edge features into table factors. The feature definition is specified in the class
 *  "FeatureGenerator".
 */

public class Trainer {

    public static ArrayList<String4Learning> string4Learning(){

        // Each string is stored as an object specifying features as table factors.
        ArrayList<String4Learning> str_list = new ArrayList<>();

        // This is a set of node feature vectors. Keys of the hash map are the feature type indices. The vectors are of
        // the same length equivalent to the string length.
        HashMap<Integer, Vector<Double>> list_node_feature; // for a single sequence
        // This is a set of transition/edge features. Keys of the outer hash map are the feature type indices. Keys of
        // the inner hash map specify the pairs/sets of node positions.
        HashMap<Integer, HashMap<Integer[], Vector<Double>>> list_edge_feature; // for a single sequence
        // This is a label vector where each entry is a label index.
        ArrayList<Integer> label_vec;   // for a single sequence

        int num_sample; // number of strings
        int num_label; // number of labels (number of possible outcomes of variables)
        int num_node_feature_type;  // number of node feature types
        int num_edge_feature_type;  // number of edge feature types
        int[] len_string; // length of each string


        // For each training sample, construct a factor graph via variables. and a list of table factors to specify edge
        // and node features.
        for(int idx_sample=0; idx_sample<num_sample; idx_sample++){

            // Wrap the data into "list_node_feature" and "list_edge_feature".
            // To-do: decide a proper data structure in the class "FeatureGenerator".

            ArrayList<Factor> factorList = new ArrayList<>();   // list of table factors for the current string

            // Declare variables.
            // Do we need to set different number of outcomes for different node variables?
            Variable[] allVars = new Variable[len_string[idx_sample]];
            for(int i=0; i<len_string[idx_sample]; i++){
                allVars[i] = new Variable(num_label);
            }

            // Add node features.
            for(int i=0; i<len_string[idx_sample]; i++){
                for(int j=0; j<num_node_feature_type; j++){
                    double[] feature_value_arr = new double[num_label];
                    // Node features and transition(edge) features are indexed separately.
                    Vector<Double> feature_vector = list_node_feature.get(j);   // for the j-th feature type
                    for(int k=0; k<num_label; k++){
                        feature_value_arr[k] = feature_vector.get(i);
                    }
                    Factor ptl = LogTableFactor.makeFromValues(new Variable[] {allVars[i]}, feature_value_arr);
                    factorList.add(ptl);
                }
            }

            // Add edge features.
            for(int i=0; i<num_edge_feature_type; i++){
                HashMap<Integer[], Vector<Double>> feature_map = list_edge_feature.get(i); // for the i-th feature type
                for(Integer[] node_set: feature_map.keySet()){
                    // Get the feature vector and store it in an array.
                    Vector<Double> feature_vector = feature_map.get(node_set);
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

    public static void main (){
        // To-do: main function for training.
    }

}
