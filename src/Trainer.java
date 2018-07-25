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

        ArrayList<Vector> list_feature_vec;
        ArrayList<Integer> label_vec;

        // For each training sample, construct a factor graph and a list of factors where the list of factors stores
        // edge and node features.



    }


}
