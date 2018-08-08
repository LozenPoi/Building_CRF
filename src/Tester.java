import edu.umass.cs.mallet.grmm.inference.Inferencer;
import edu.umass.cs.mallet.grmm.inference.TRP;
import edu.umass.cs.mallet.grmm.types.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Vector;

public class Tester {

    public double acc(){
        //
    }

    public double doTesting(ArrayList<FactorGraph> testGraphSet, ArrayList<Assignment>testAssignment){
        FactorGraph graph;
        Assignment assin, guess;
        AssignmentIterator it;
        Factor ptl;
        Variable variable;
        int varSize, var, parID = 0;
        int[] prediction;
        double max, correct = 0, total = 0, TP = 0, LL = 0;
        Vector<Double> acc = new Vector<>();
        String filename;

        Inferencer map_infer = TRP.createForMaxProduct();
        //Inferencer map_infer = LoopyBP.createForMaxProduct();
        for(int sampleID=0; sampleID<m_trainSampleSet.size(); sampleID++) {
            graph = testGraphSet.get(sampleID);
            assin = testAssignment.get(sampleID);
            varSize = graph.numVariables();
            correct = 0;
            prediction = new int[varSize];
            map_infer.computeMarginals(graph);  //begin to collect the expectations
            for(var=1; var<varSize; var++) {
                //retrieve the MAP configuration
                variable = graph.get(var);
                ptl = map_infer.lookupMarginal( variable );
                max = -Double.MAX_VALUE;
                for (it = ptl.assignmentIterator (); it.hasNext (); it.next()) {
                    if (ptl.value(it)>max) {
                        max = ptl.value(it);
                        parID = it.indexOfCurrentAssn();
                    }
                }
                prediction[var] = parID;
                //evaluate the performance
                if( parID == assin.get(variable) )
                    correct++;
            }

            guess = new Assignment(graph, prediction);
            if ( map_infer.lookupLogJoint(guess) > map_infer.lookupLogJoint(assin) )
                LL++;

            prediction[0] = -1;
            if (resultpath!=null){
                filename = m_trainSampleSet.get(sampleID).threadid.get(0);
                outPrediction(resultpath + filename + ".res", prediction);
            }

            acc.add(correct/(varSize-1));
            TP += correct;
            total += varSize-1;
        }

        correct = 0;
        for(parID=0; parID<acc.size(); parID++)
            correct += acc.get(parID).doubleValue();
        System.out.println("Micro accuracy " + TP/total);
        System.out.println("Macro accuracy " + correct/acc.size());
        System.out.println(LL/acc.size() + " percentage threads have better configuration in likelihood!");
        return TP/total;
    }

    public void main(){
        // Read data files.

    }

}
