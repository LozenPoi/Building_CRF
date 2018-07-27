import java.util.ArrayList;
import java.util.HashMap;

/**
 * This is a class defining the feature types, and processing data to create dictionaries and return feature vectors.
 */

public class FeatureGenerator {

    public HashMap<Integer,String> dict_token;          // a dictionary for characters (tokens) in a string
    public HashMap<Integer,String> dict_label;          // a dictionary for labels
    public HashMap<Integer,String> dict_node_feature;   // a dictionary for node feature types
    public HashMap<Integer,String> dict_edge_feature;   // a dictionary for node feature types

    // Initialize this object.
    public FeatureGenerator(){
        dict_token = new HashMap<>();
        dict_label = new HashMap<>();
        dict_node_feature = new HashMap<>();
        dict_edge_feature = new HashMap<>();
        // Assign an ID to each feature type.
        dict_node_feature.put(1,"token");
        dict_edge_feature.put(1,"B followed by I");
    }

    // Read data through files.
    public void readData(){}

    // Get the node features for all node feature types.
    public void getNodeFeature(){}

    // Get the edge features for all edge feature types.
    public void getEdgeFeature(){}

    // Update the token dictionary.
    public void update_token(){}

    // Update the label dictionary.
    public void update_label(){}


    // This is a node feature which is the character itself.
    private int currentToken(char token){
        int index;
        if(dict_token.containsValue(token)){
            // get the index
        }
        else{
            // update the dictionary and get the index
        }
        return index;
    }

    // This is an edge feature which indicates label "B" followed by label "I".
    private boolean B_to_I(){
        //
    }


}
