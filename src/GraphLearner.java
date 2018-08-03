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
 * @author Hongning Wang (wang296@illinois.edu)
 * Function: Learning model parameters by l-BFGS
 */

public class GraphLearner implements Maximizable.ByGradient{
    private double[] m_weights; //weights for each feature, to be optimized
    private Map<Integer, Boolean> m_mask; //turn on/off the features to learn
    private double[] m_constraints; //observed counts for each feature (over cliques) in the training sample set (x, y)
    private double[] m_exptectations; //expected counts for each feature (over cliques) based on the training sample set (X)

    private int[] m_foldAssign;
    private int m_foldID;

    Maximizer.ByGradient m_maxer = new LimitedMemoryBFGS();//gradient based optimizer

    ArrayList<String4Learning> m_trainSampleSet = null; // training sample (factor, feaType, Y)
    ArrayList<FactorGraph> m_trainGraphSet = null;
    ArrayList<Assignment> m_trainAssignment = null;

    TreeMap<Integer, Integer> m_featureMap;

    Inferencer m_infer; //inferencer for marginal computation

    boolean m_scaling;
    boolean m_updated;
    boolean m_trained;
    double m_lambda;
    double m_oldLikelihood;
    Random m_rand = new Random();

    BufferedWriter m_writer;

    GraphLearner(ArrayList<String4Learning> traininglist){
        m_infer = new LoopyBP(50); //TRP(); //

        int featureDim = setTrainingSet(traininglist);
        m_weights = new double[featureDim];
        m_mask = null;
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

    private int setTrainingSet(ArrayList<String4Learning> traininglist){
        m_trainSampleSet = traininglist;

        m_featureMap = new TreeMap<Integer, Integer>();
        for(String4Learning sample : traininglist)
        {
            for(Integer feature : sample.featureType)
            {
                if ( m_featureMap.containsKey(feature) == false )
                    m_featureMap.put(feature, new Integer(m_featureMap.size()));
            }
        }
        return m_featureMap.size();
    }

    private void allocateFoldAssignment(int fold)
    {
        m_foldAssign = new int[m_trainSampleSet.size()];
        for(int i=0; i<m_trainSampleSet.size(); i++){
            m_foldAssign[i] = m_rand.nextInt(fold);
        }
    }

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

    public void LoadWeights(String filename){
        try {
            BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));
            String tmpTxt;
            String[] feature;
            int feaPtx;
            Integer fea;
            while( (tmpTxt=reader.readLine()) != null ){
                feature = tmpTxt.split(" : ");
                fea = new Integer(feature[0]);
                if( m_featureMap.containsKey(fea) )
                {
                    feaPtx = m_featureMap.get(fea).intValue();
                    m_weights[feaPtx] = Double.valueOf(feature[1]);
                }
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void SaveWeights(String filename){
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(filename)));//append mode
            Map<Integer, Double> weights = getWeights();
            for(Integer fea : weights.keySet())
                writer.write(fea.toString() + " : " + weights.get(fea).toString() + "\n");
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void setMask(Map<Integer, Boolean> masks)
    {
        m_mask = masks;
    }

    @Override
    public int getNumParameters() {
        return m_weights.length;
    }

    @Override
    public double getParameter(int index) {
        if ( index<m_weights.length)
            return m_weights[index];
        else
            return 0;
    }

    @Override
    public void getParameters(double[] buffer) {
        if ( buffer.length != m_weights.length )
            buffer = new double[m_weights.length];
        System.arraycopy(m_weights, 0, buffer, 0, m_weights.length);
    }

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
        String4Learning sample = null;
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

    void initWeight(){
        for(int i=0; i<m_weights.length; i++)
            m_weights[i] = 1.0; //rand.nextDouble();
    }

    void initialization(boolean initWeight){
        FactorGraph graph = null;
        String4Learning sample = null;
        Assignment assign = null;
        Factor factor = null;
        int[] assignment;
        int i, feaID;
        double feaValue;

        //build the initial factor graph
        if( initWeight )
            initWeight();
        buildFactorGraphs();

        //collect the feature counts in the training set
        m_trainAssignment = new ArrayList<Assignment>(m_trainSampleSet.size());
        for(int sampleID=0; sampleID<m_trainSampleSet.size(); sampleID++){
            graph = m_trainGraphSet.get(sampleID);
            sample = m_trainSampleSet.get(sampleID);

            //get the answer's assignment over the graph
            assignment = new int[sample.parent.size()];
            for(i=0; i<sample.parent.size(); i++)
                assignment[i] = sample.parent.get(i).intValue();
            assign = new Assignment(graph, assignment);
            m_trainAssignment.add(assign);

            if( m_foldAssign != null && m_foldAssign[sampleID] == m_foldID )
                continue;//for cross validation

            for(i=0; i<sample.factorList.size(); i++)
            {
                if ( m_mask != null && m_mask.containsKey(sample.featureType.get(i))
                        && m_mask.get(sample.featureType.get(i)).booleanValue() == false )
                    continue;

                factor = sample.factorList.get(i);
                feaID = m_featureMap.get( sample.featureType.get(i) ).intValue();
                feaValue = factor.logValue(assign);//retrieve the assignment from the subset
                m_constraints[feaID] += feaValue;
            }
        }
        System.out.println("Finish collecting sufficient statistics...");

    }

    void buildFactorGraphs(){
        //convert and cache the factors in each thread into a factor graph
        boolean init = (m_trainGraphSet == null);
        if ( init == true )//init for the first time
            m_trainGraphSet = new ArrayList<FactorGraph>(m_trainSampleSet.size());

        FactorGraph threadGraph = null;
        String4Learning tmpThread = null;
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

    public void outPrediction(String filename, int[] prediction)
    {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(filename)));//append mode
            for(int i=0; i<prediction.length; i++){
                writer.write(i + "\t" + prediction[i] + "\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public double doTesting(String resultpath){
        FactorGraph graph = null;
        Assignment assin = null, guess = null;
        AssignmentIterator it;
        Factor ptl;
        Variable variable;
        int varSize, var, parID = 0;
        int[] prediction;
        double max, correct = 0, total = 0, TP = 0, LL = 0;
        Vector<Double> acc = new Vector<Double>();
        String filename;

        if( m_trainGraphSet == null )
            initialization(false);//solely for testing purpose

        Inferencer map_infer = TRP.createForMaxProduct();
        //Inferencer map_infer = LoopyBP.createForMaxProduct();
        for(int sampleID=0; sampleID<m_trainSampleSet.size(); sampleID++)
        {
            if ( m_foldAssign != null && m_foldAssign[sampleID] != m_foldID )
                continue;

            graph = m_trainGraphSet.get(sampleID);
            assin = m_trainAssignment.get(sampleID);
            if ( (varSize = graph.numVariables()) < 2 )
                continue;

            correct = 0;
            prediction = new int[varSize];
            map_infer.computeMarginals(graph);//begin to collect the expectations
            for(var=1; var<varSize; var++)
            {
                //retrieve the MAP configuration
                variable = graph.get(var);
                ptl = map_infer.lookupMarginal( variable );
                max = -Double.MAX_VALUE;
                for (it = ptl.assignmentIterator (); it.hasNext (); it.next())
                {
                    if (ptl.value(it)>max)
                    {
                        max = ptl.value(it);
                        parID = it.indexOfCurrentAssn ();
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

    public double doTraining(int maxIter)
    {
        initialization(true);//build the initial factor graphs and collect the constraints from data
        double oldLikelihood = getValue(), likelihood; // initial likelihood
        try
        {
            if ( m_maxer.maximize(this, maxIter) == false )
            {//if failed, try it again
                System.err.println("Optimizer fails to converge!");
                ((LimitedMemoryBFGS)m_maxer).reset();
                m_maxer.maximize(this, maxIter);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        likelihood = getValue();
        m_trained = true;

        System.out.println("Training process start, with likelihood " + oldLikelihood);
        System.out.println("Training process finish, with likelihood " + likelihood);

        try
        {
            Map<Integer, Double> weights = getWeights();
            for(Integer fea : weights.keySet())
                m_writer.write(fea.toString() + " : " + weights.get(fea).toString() + " ");
            m_writer.write("\t" + likelihood + "\n");
            m_writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return likelihood;
    }

    public Vector<Double> getMeanDev(Vector<Double> stat)
    {
        Vector<Double> result = new Vector<Double>();
        double mean = 0, dev = 0;
        for(Double value : stat)
            mean += value.doubleValue();
        mean /= stat.size();
        result.add(mean);

        for(Double value : stat)
            dev += (value.doubleValue() - mean) * (value.doubleValue() - mean);
        dev = Math.sqrt(dev/stat.size());
        result.add(dev);
        return result;
    }

    public void CrossValidation(int fold)
    {
        allocateFoldAssignment(fold);
        Vector<Double> likelihood_list = new Vector<Double>(), acc_list = new Vector<Double>();

        for(m_foldID=0; m_foldID<fold; m_foldID++)
        {
            likelihood_list.add( doTraining(10) );
            acc_list.add( doTesting(null) );
        }

        Vector<Double> likelihood_result = getMeanDev(likelihood_list), acc_result = getMeanDev(acc_list);
        System.out.println("Performance of " + fold + " fold cross-validation: \n"
                + "log-likelihood " + likelihood_result.get(0) + "+/-" + likelihood_result.get(1)
                + "\naccuracy " + acc_result.get(0) + "+/-" + acc_result.get(1));
    }
}
