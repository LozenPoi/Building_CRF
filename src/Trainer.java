import java.util.ArrayList;
import java.util.Vector;

/**
 *  This is a class putting node and edge features into table factors. The feature definition is specified in the class
 *  "FeatureGenerator".
 */

public class Trainer {

    public static ArrayList<String4Learning> string4Learning(String[] feature_dict, String[] label_dict,
                                                             String train_path){

        // Each string is stored as an object specifying features as table factors.
        ArrayList<String4Learning> str_list = new ArrayList<>();

        ArrayList<Vector> list_node_feature;
        ArrayList<Vector> list_edge_feature;
        ArrayList<Integer> label_vec;

        // For each training sample, construct a factor graph via variables. and a list of table factors to specify edge
        // and node features.
        for



    }


}
