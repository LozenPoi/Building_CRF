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
        dict_node_feature.put(0,"token");
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
    private int currentToken(String token){
        int index = 0;
        if(dict_token.containsValue(token)){
            for(Integer idx: dict_token.keySet()){
                if(dict_token.get(idx).equals(token)){
                    index = idx;
                }
            }
        }
        else{
            index = dict_token.size();
            dict_token.put(index,token);
        }
        return index;
    }

    // This is an edge feature which indicates label "B" followed by label "I".
    private boolean B_to_I(String label_previous, String label_current){
        if(label_previous.substring(2).equals(label_current.substring(2))){
            return (label_previous.charAt(0) == 'b') && (label_current.charAt(0) == 'i');
        }
        else {
            return false;
        }
    }


}
