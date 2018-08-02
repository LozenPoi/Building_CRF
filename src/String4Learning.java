import edu.umass.cs.mallet.grmm.types.Factor;

import java.util.List;

// This is a class storing strings as feature vectors.
public class String4Learning {

    List<Factor> factorList;
    List<Integer> featureType;  //feature type correspond to factors

    public String4Learning(List<Factor> factorList, List<Integer> featureType) {
        this.factorList = factorList;
        this.featureType = featureType;
    }

    public List<Factor> getFactorList() {
        return factorList;
    }
    public void setFactorList(List<Factor> factorList) {
        this.factorList = factorList;
    }
    public List<Integer> getFeatureType() {
        return featureType;
    }
    public void setFeatureType(List<Integer> featureType) {
        this.featureType = featureType;
    }

}
