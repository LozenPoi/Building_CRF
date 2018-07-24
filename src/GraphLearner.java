import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.Vector;
import java.util.Map.Entry;

import edu.umass.cs.mallet.base.maximize.LimitedMemoryBFGS;
import edu.umass.cs.mallet.base.maximize.Maximizable;
import edu.umass.cs.mallet.base.maximize.Maximizer;
import edu.umass.cs.mallet.grmm.inference.Inferencer;
import edu.umass.cs.mallet.grmm.inference.LoopyBP;
import edu.umass.cs.mallet.grmm.inference.TRP;
import edu.umass.cs.mallet.grmm.types.Assignment;
import edu.umass.cs.mallet.grmm.types.AssignmentIterator;
import edu.umass.cs.mallet.grmm.types.Factor;
import edu.umass.cs.mallet.grmm.types.FactorGraph;
import edu.umass.cs.mallet.grmm.types.VarSet;
import edu.umass.cs.mallet.grmm.types.Variable;

/**
 * This is a graphical model for conditional random field, using a single weight vector for all classes.
 * It learns and stores model parameters where the factor graph is specified outside of this class.
 * The training sample is in the form of feature map where keys stand for feature indices.
 * @author Hongning Wang and Zheng Luo
 */

public class GraphLearner implements Maximizable.ByGradient {

    private double[] m_weights; // weights for features (analog to logistic regression) to be optimized
    private double[] m_constraints; // observed counts for each feature (over cliques) in the training sample set (x, y)
    private double[] m_exptectations; // expected counts for each feature (over cliques) based on the training sample set (X)

    private int[] m_foldAssign;
    private int m_foldID;

    Maximizer.ByGradient m_maxer = new LimitedMemoryBFGS(); // gradient based optimizer

    ArrayList<Thread4Learning> m_trainSampleSet = null; // training sample (factor, feaType, Y)
    ArrayList<FactorGraph> m_trainGraphSet = null;
    ArrayList<Assignment> m_trainAssignment = null;
    ArrayList<String> m_trainIDs = null;

    TreeMap<Integer, Integer> m_featureMap;

    Inferencer m_infer; // inferencer for marginal computation

    boolean m_scaling;
    boolean m_updated;
    boolean m_trained;
    double m_lambda;
    double m_oldLikelihood;
    Random m_rand = new Random();

    BufferedWriter m_writer;

    GraphLearner(ArrayList<Thread4Learning> traininglist){

        m_infer = new LoopyBP(50); // This is the inferencer for loop graph.

        int featureDim = setTrainingSet(traininglist);
        m_weights = new double[featureDim];
        m_constraints = new double[featureDim];
        m_exptectations = new double[featureDim];

        //training parameters
        m_scaling = false;
        m_updated = true;
        m_trained = false;
        m_rand = new Random();
        m_foldAssign = null;

        m_lambda = 2.0;//L2 regularization
        m_oldLikelihood = -Double.MAX_EXPONENT;//init value

        try {
            m_writer = new BufferedWriter(new FileWriter(new File("trace.dat")));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Get the number of weights (the length of the weighting vector).
    @Override
    public int getNumParameters() {
        return m_weights.length;
    }

    // Get a specific weight (an entry).
    @Override
    public double getParameter(int index) {
        if ( index<m_weights.length)
            return m_weights[index];
        else
            return 0;
    }

    // Copy the whole weighting vector to a buffer (just want to implement an abstract method?).
    @Override
    public void getParameters(double[] buffer) {
        if ( buffer.length != m_weights.length )
            buffer = new double[m_weights.length];
        System.arraycopy(m_weights, 0, buffer, 0, m_weights.length);
    }


    // Change a specific entry of the weighting vector (or extend the weighting vector).
    @Override
    public void setParameter(int index, double value) {
        if ( index<m_weights.length )
            m_weights[index] = value;
        else{
            double[] weights = new double[index+1];
            System.arraycopy(weights, 0, m_weights, 0, m_weights.length);
            weights[index] = value;
            m_weights = weights;
        }
    }

    // Set up the whole weighting vector specified by "params".
    @Override
    public void setParameters(double[] params) {
        if( params.length != m_weights.length )
            m_weights = new double[params.length];

        Map<Integer, Double> weights = getWeights();
        try {
            for(Integer fea : weights.keySet())
                m_writer.write(fea.toString() + " : " + weights.get(fea).toString() + " ");
            m_writer.write("\t" + m_oldLikelihood + "\n");
            m_writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.arraycopy(params, 0, m_weights, 0, m_weights.length);
        m_updated = true;
    }

    //
    @Override
    public double getValue() {
        if ( m_updated )
        {
            buildFactorGraphs();
            m_updated = false;
        }
        else
            return m_oldLikelihood;

        double tmp;
        FactorGraph graph = null;
        Assignment assign = null;

        m_oldLikelihood = 0;
        for(int feaID=0; feaID<m_weights.length; feaID++)
            m_oldLikelihood -= m_weights[feaID] * m_weights[feaID];
        m_oldLikelihood *= m_lambda / 2;//L2 penalty

        double scale = m_scaling ? (1.0/m_trainSampleSet.size()) : 1.0;
        for(int threadID=0; threadID<m_trainGraphSet.size(); threadID++)
        {
            if( m_foldAssign !=null && m_foldAssign[threadID] == m_foldID )
                continue;//for cross validation

            assign = m_trainAssignment.get(threadID);
            graph = m_trainGraphSet.get(threadID);
            m_infer.computeMarginals(graph);
            tmp = m_infer.lookupLogJoint(assign);
            if( Double.isNaN(tmp) == true || tmp>0 )
                System.err.println("In thread " + m_trainIDs.get(threadID) + " likelihood failed with " + tmp + "!");
            else
                m_oldLikelihood += tmp * scale;
        }

        System.out.println("[Info]Log-likelihood " + m_oldLikelihood);
        return m_oldLikelihood;//negative log-likelihood or log-likelihood?
    }

    @Override
    public void getValueGradient(double[] buffer) {
        FactorGraph graph = null;
        Factor factor = null, ptl = null;
        Thread4Learning sample = null;
        int feaID;

        double feaValue, prob;

        for(feaID=0; feaID<m_exptectations.length; feaID++)
            m_exptectations[feaID] = 0;//clear the SS

        for(int sampleID=0; sampleID<m_trainSampleSet.size(); sampleID++)
        {
            if( m_foldAssign !=null && m_foldAssign[sampleID] == m_foldID )
                continue;//for cross validation

            graph = m_trainGraphSet.get(sampleID);
            sample = m_trainSampleSet.get(sampleID);

            m_infer.computeMarginals(graph);//begin to collect the expectations
            for(int index=0; index<sample.factorList.size(); index++)
            {
                if ( m_mask != null && m_mask.containsKey(sample.featureType.get(index))
                        && m_mask.get(sample.featureType.get(index)).booleanValue() == false )
                    continue;

                factor = sample.factorList.get(index);
                ptl = m_infer.lookupMarginal(factor.varSet());
                feaID = m_featureMap.get( sample.featureType.get(index) ).intValue();

                AssignmentIterator assnIt = ptl.assignmentIterator ();
                while (assnIt.hasNext ()) {
                    feaValue = factor.logValue(assnIt);//feature value;
                    prob = ptl.value (assnIt);//get the marginal probability for this local configuration
                    m_exptectations[feaID] += feaValue * prob;
                    assnIt.advance ();
                }
            }
        }

        double scale = m_scaling ? (1.0/m_trainSampleSet.size()) : 1.0;
        for(feaID=0; feaID<m_weights.length; feaID++){
            buffer[feaID] = scale * (m_constraints[feaID] - m_exptectations[feaID]) - (m_weights[feaID] * m_lambda);
        }
    }

    // Get a copy of weights in the form of tree map where keys are feature indices.
    public Map<Integer, Double> getWeights(){
        Map<Integer, Double> weights = new TreeMap<Integer, Double>();
        Iterator<Entry<Integer, Integer>> it = m_featureMap.entrySet().iterator();
        Map.Entry<Integer, Integer> pairs;
        while (it.hasNext())
        {
            pairs = (Map.Entry<Integer, Integer>)it.next();
            weights.put(pairs.getKey(), new Double(m_weights[pairs.getValue().intValue()]));
        }
        return weights;
    }

    public void buildFactorGraphs(){
        //convert and cache the factors in each thread into a factor graph
        boolean init = (m_trainGraphSet == null);
        if ( init == true )//init for the first time
            m_trainGraphSet = new ArrayList<FactorGraph>(m_trainSampleSet.size());

        FactorGraph threadGraph = null;
        Thread4Learning tmpThread = null;
        Factor factor = null;
        VarSet clique = null;
        int index, feaID, threadID;

        HashMap<VarSet, Integer> factorIndex = new HashMap<VarSet, Integer>();
        Vector<Factor> factorList = new Vector<Factor>();

        for(threadID=0; threadID<m_trainSampleSet.size(); threadID++)
        {
            tmpThread = m_trainSampleSet.get(threadID);
            if ( init )
                threadGraph = new FactorGraph();
            else
            {
                threadGraph = m_trainGraphSet.get(threadID);
                threadGraph.clear();//is it safe?
            }

            factorIndex.clear();
            factorList.clear();
            for(index=0; index<tmpThread.factorList.size(); index++)
            {
                if ( m_mask != null && m_mask.containsKey(tmpThread.featureType.get(index))
                        && m_mask.get(tmpThread.featureType.get(index)).booleanValue() == false )
                    continue;

                factor = tmpThread.factorList.get(index);
                Factor copy = factor.duplicate();
                feaID = m_featureMap.get( tmpThread.featureType.get(index) ).intValue();

                copy.exponentiate( m_weights[feaID] );//potential = feature * weight
                clique = copy.varSet();//to deal with factors defined over the same clique
                if( factorIndex.containsKey(clique) )
                {
                    feaID = factorIndex.get(clique).intValue();
                    factor = factorList.get(feaID);
                    factor.multiplyBy(copy);
                }
                else
                {
                    factorIndex.put(clique, new Integer(factorList.size()));
                    factorList.add(copy);
                }
            }

            //construct the graph
            for(index=0; index<factorList.size(); index++)
                threadGraph.addFactor(factorList.get(index));

            if ( init )
                m_trainGraphSet.add(threadGraph);
        }

        if ( init )
            System.out.println("Finish building " + m_trainGraphSet.size() + "factor graphs...");
    }

    // Update the model parameters using a sample.
    // This is useful for active learning (the training set is updated by adding new samples into it).
    public void updateModel(TreeMap<Integer, Integer> new_featureMap){
        //
    }

}
