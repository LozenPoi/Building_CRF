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
        dict_node_feature.put(0,"currentToken");
        dict_edge_feature.put(0,"B_to_I");
    }

    // Get the node features for all node feature types.
    public HashMap<Integer,ArrayList<Double>> getNodeFeature(String sample){
        HashMap<Integer,ArrayList<Double>> feature_vectors = new HashMap<>();
        // Get the current token (feature ID: 0).
        ArrayList<Double> current_token = new ArrayList<>();
        // Get if it is a digit (feature ID: 1).
        ArrayList<Double> is_digit = new ArrayList<>();
        for(int i=0; i<sample.length(); i++){
            current_token.add((double)currentToken(Character.toString(sample.charAt(i))));
            if(isDigit(Character.toString(sample.charAt(i)))) {
                is_digit.add(1.0);
            }else{
                is_digit.add(0.0);
            }
        }
        feature_vectors.put(0,current_token);
        feature_vectors.put(1,is_digit);
        return feature_vectors;
    }

    // Update the label dictionary.
    public int update_label(String label){
        int size;
        if(!dict_label.containsValue(label)){
            size = dict_label.size();
            dict_label.put(size,label);
        }else{
            size = getLabelIdx(label);
        }
        return size;
    }

    // Get a label index.
    public int getLabelIdx(String label){
        for(Integer i: dict_label.keySet()){
            if(dict_label.get(i).equals(label)){
                return i;
            }
        }
        return -1;  // This indicates that the label is not in the dictionary.
    }

    // This is a node feature which is the character itself.
    public int currentToken(String token){
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
            dict_token.put(index,token);    // Update token dictionary.
        }
        return index;
    }

    // Check if a character (token) is a digit.
    private boolean isDigit(String token){
        char token_char = token.charAt(0);
        return Character.isDigit(token_char);
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
