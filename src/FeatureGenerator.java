import java.util.ArrayList;
import java.util.HashMap;

/**
 * This is a class defining the feature types, and processing data to create dictionaries and return feature vectors.
 */

public class FeatureGenerator {

    public HashMap<Integer,String> dict_token;   // a dictionary for characters (tokens) in a string
    public HashMap<Integer,String> dict_label;   // a dictionary for labels
    public HashMap<Integer,String> dict_feature; // a dictionary for feature types

    // Initialize this object.
    public FeatureGenerator(){
        dict_token = new HashMap<>();
        dict_label = new HashMap<>();
        dict_feature = new HashMap<>();
    }

    // Read data through files.
    public void readData(){}

    // Get the node features for all node feature types.
    public void getNodeFeature(){}

    // Get the edge features for all edge feature types.
    public void getEdgeFeature(){}

    //


}
