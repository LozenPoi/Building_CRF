package utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

public class Utils {
	
	public static double MAX_VALUE = 1e5;
	
	//Find the max value's index of an array, return Index of the maximum.
	public static int maxOfArrayIndex(double[] probs){
		return maxOfArrayIndex(probs, probs.length);
	}
	
	public static int maxOfArrayIndex(double[] probs, int length){
		int maxIndex = 0;
		double maxValue = probs[0];
		for(int i = 1; i < length; i++){
			if(probs[i] > maxValue){
				maxValue = probs[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	public static int minOfArrayIndex(double[] probs, int length){
		int minIndex = 0;
		double minValue = probs[0];
		for(int i = 1; i < length; i++){
			if(probs[i] < minValue){
				minValue = probs[i];
				minIndex = i;
			}
		}
		return minIndex;
	}
	
	//Calculate the sum of a column in an array.
	public static double sumOfRow(double[][] mat, int i){
		return sumOfArray(mat[i]);
	}
	
	//Calculate the sum of a row in an array.
	public static double sumOfColumn(double[][] mat, int i){
		double sum = 0;
		for(int j = 0; j < mat.length; j++){
			sum += mat[j][i];
		}
		return sum;
	}
	
	//Calculate the sum of a column in an array.
	public static int sumOfRow(int[][] mat, int i){
		return sumOfArray(mat[i]);
	}
	
	//Calculate the sum of a row in an array.
	public static int sumOfColumn(int[][] mat, int i){
		int sum = 0;
		for(int j = 0; j < mat.length; j++){
			sum += mat[j][i];
		}
		return sum;
	}
	
	public static double entropy(double[] prob, boolean logScale) {
		double ent = 0;
		for(double p:prob) {
			if (logScale)
				ent += Math.exp(p) * p;
			else
				ent += Math.log(p) * p;
		}
		return -ent;
	}
	
	//Find the max value's index of an array, return Value of the maximum.
	public static double maxOfArrayValue(double[] probs){
		return probs[maxOfArrayIndex(probs)];
	}
	
	//This function is used to calculate the log of the sum of several values.
	public static double logSumOfExponentials(double[] xs){
		if(xs.length == 1){
			return xs[0];
		}
		
		double max = maxOfArrayValue(xs);
		double sum = 0.0;
		for (int i = 0; i < xs.length; i++) {
			if (!Double.isInfinite(xs[i])) 
				sum += Math.exp(xs[i] - max);
		}
		
		if (sum==0)
			return max;
		return Math.log(sum) + max;
	}
	
	public static double logSum(double log_a, double log_b) {
		if (Double.isInfinite(log_a))
			return log_b;
		else if (Double.isInfinite(log_b))
			return log_a;
		else if (log_a < log_b)
			return log_b+Math.log(1 + Math.exp(log_a-log_b));
		else
			return log_a+Math.log(1 + Math.exp(log_b-log_a));
	}
	
	//The function is used to calculate the sum of log of two arrays.
	public static double sumLog(double[] probs, double[] values){
		double result = 0;
		if(probs.length == values.length){
			for(int i = 0; i < probs.length; i++){
				result += values[i] * Math.log(probs[i]);
			}
		} else{
			System.out.println("log sum fails due to the lenghts of two arrars are not matched!!");
		}
		return result;
	}

	public static double dotProduct(double[] beta, int[] f, int offset){
		double sum = beta[offset];
        for(int i = 0; i < f.length; i++) {
			int idx = i + offset + 1;
			sum += beta[idx] * f[i];
		}
		return sum;
	}

	public static double dotProduct(double[] a, double[] b) {
		if (a.length != b.length)
			return Double.NaN;
		double sum = 0;
		for(int i=0; i<a.length; i++)
			sum += a[i] * b[i];
		return sum;
	}
	
	public static double L2Norm(double[] a) {
		return Math.sqrt(dotProduct(a,a));
	}
	
	public static double L2Norm(double[] a, double[] b) {
		if (a.length != b.length)
			return Double.NaN;
		double diff=0;
		for(int i=0; i<a.length; i++)
			diff += (a[i]-b[i])*(a[i]-b[i]);
		return diff;
	}
	
	//Logistic function: 1.0 / (1.0 + exp(-wf))
	public static double logistic(double[] fv, double[] w){
		double sum = w[0];//start from bias term
		for(int i = 0; i < fv.length; i++)
			sum += fv[i] * w[1+i];
		return logistic(sum);
	}
	
	public static double logistic(double v) {
		return 1.0 / (1.0 + Math.exp(-v));
	}
	
	//The function defines the sum of an array.
	public static int sumOfArray(int[] a){
		int sum = 0;
		for (int i: a)
			sum += i;
		return sum;
	}
	
	//The function defines the sum of an array.
	public static double sumOfArray(double[] a) {
		double sum = 0;
		for (double i : a)
			sum += i;
		return sum;
	}
	
	public static double[] diff(double[] a, double[] b) {
		if (a.length != b.length)
			return null;
		
		double[] diff = new double[a.length];
		boolean nonzero = false;
		for(int i=0; i<a.length; i++) {
			diff[i] = a[i] - b[i];
			if (Math.abs(diff[i])>1e-10)
				nonzero = true;
		}
		return nonzero?diff:null;
	}
	
	public static void scaleArray(double[] a, double b) {
		for (int i=0; i<a.length; i++)
			a[i] *= b;
	}
	
	public static void scaleArray(double[] a, double[] b, double scale) {
		for (int i=0; i<a.length; i++)
			a[i] += b[i] * scale;
	}
	
	public static void setArray(double[] a, double[] b, double scale) {
		for (int i=0; i<a.length; i++)
			a[i] = b[i] * scale;
	}
	
	static public void add2Array(double[] vct, double[] add, double weight){
		if (vct.length != add.length)
			return;
		for(int i=0; i<vct.length; i++)
			vct[i] += weight * add[i];
	}
	
	//L1 normalization: fsValue/sum(abs(fsValue))
	static public double sumOfFeaturesL1(_SparseFeature[] fs) {
		double sum = 0;
		for (_SparseFeature feature: fs)
			sum += Math.abs(feature.getValue());
		return sum;
	}
	
	//Set the normalized value back to the sparse feature.
	static public void L1Normalization(_SparseFeature[] fs) {
		double sum = sumOfFeaturesL1(fs);
		if (sum>0) {
			//L1 length normalization
			for(_SparseFeature f:fs){
				double normValue = f.getValue()/sum;
				f.setValue(normValue);
			}
		} else{
			for(_SparseFeature f: fs){
				f.setValue(0.0);
			}
		}
	}
	
	//L1 normalization
	static public void L1Normalization(double[] v) {
		double sum = sumOfArray(v);
		if (sum>0) {			
			for(int i=0; i<v.length; i++){
				v[i] /= sum;
			}
		}
	}
	
	//L2 normalization: fsValue/sqrt(sum of fsValue*fsValue)
	static public double sumOfFeaturesL2(_SparseFeature[] fs) {
		if(fs == null) 
			return 0;
		
		double sum = 0;
		for (_SparseFeature feature: fs){
			double value = feature.getValue();
			sum += value * value;
		}
		return Math.sqrt(sum);
	}
	
	static public void L2Normalization(_SparseFeature[] fs) {
		double sum = sumOfFeaturesL2(fs);
		if (sum>0) {			
			for(_SparseFeature f: fs){
				double normValue = f.getValue()/sum;
				f.setValue(normValue);
			}
		}
		else{
			for(_SparseFeature f: fs){
				f.setValue(0.0);
			}
		}
	}
	
	public static double jaccard(_SparseFeature[] spVct1, _SparseFeature[] spVct2) {
		if (spVct1==null || spVct2==null)
			return 0; // What is the minimal value of similarity?
		
		double overlap = 0;
		int pointer1 = 0, pointer2 = 0;
		while (pointer1 < spVct1.length && pointer2 < spVct2.length) {
			_SparseFeature temp1 = spVct1[pointer1];
			_SparseFeature temp2 = spVct2[pointer2];
			if (temp1.getIndex() == temp2.getIndex()) {
				overlap ++;
				pointer1++;
				pointer2++;
			} else if (temp1.getIndex() > temp2.getIndex())
				pointer2++;
			else
				pointer1++;
		}
		return overlap/(spVct1.length + spVct2.length);
	}
	
	public static double cosine(_SparseFeature[] spVct1, _SparseFeature[] spVct2) {
		double spVct1L2 = sumOfFeaturesL2(spVct1), spVct2L2 = sumOfFeaturesL2(spVct2);
		if (spVct1L2==0 || spVct2L2==0)
			return 0;
		else
			return calculateSimilarity(spVct1, spVct2) / spVct1L2 / spVct2L2;
	}
	
	public static double cosine(double[] a, double[] b) {
		return dotProduct(a, b) / L2Norm(a) / L2Norm(b);
	}
	
	//Calculate the similarity between two sparse vectors.
	public static double calculateSimilarity(_SparseFeature[] spVct1, _SparseFeature[] spVct2) {
		if (spVct1==null || spVct2==null)
			return 0; // What is the minimal value of similarity?
		
		double similarity = 0;
		int pointer1 = 0, pointer2 = 0;
		while (pointer1 < spVct1.length && pointer2 < spVct2.length) {
			_SparseFeature temp1 = spVct1[pointer1];
			_SparseFeature temp2 = spVct2[pointer2];
			if (temp1.getIndex() == temp2.getIndex()) {
				similarity += temp1.getValue() * temp2.getValue();
				pointer1++;
				pointer2++;
			} else if (temp1.getIndex() > temp2.getIndex())
				pointer2++;
			else
				pointer1++;
		}
		return similarity;
	}
	
	static public boolean isNumber(String token) {
		return token.matches("\\d+");
	}
	
	static public void randomize(double[] pros, double beta) {
        double total = 0;
        for (int i = 0; i < pros.length; i++) {
            pros[i] = beta + Math.random();//to avoid zero probability
            total += pros[i];
        }

        //normalize
        for (int i = 0; i < pros.length; i++)
            pros[i] /= total;
    }
	
	static public String formatArray(double [] array) {
		StringBuffer buffer = new StringBuffer(256);
		for(int i=0;i<array.length;i++)
			if (i==0)
				buffer.append(Double.toString(array[i]));
			else
				buffer.append("," + Double.toString(array[i]));
		return String.format("(%s)", buffer.toString());
	}
	
	static public _SparseFeature[] createSpVct(double[] denseFv) {
		ArrayList<_SparseFeature> spVct = new ArrayList<_SparseFeature>();
		for(int i=0; i<denseFv.length; i++) {
			if (denseFv[i]!=0)
				spVct.add(new _SparseFeature(i, denseFv[i]));
		}
		return spVct.toArray(new _SparseFeature[spVct.size()]);
	}
	
	static public _SparseFeature[] createSpVct(HashMap<Integer, Double> vct) {
		_SparseFeature[] spVct = new _SparseFeature[vct.size()];
		
		int i = 0;
		Iterator<Entry<Integer, Double>> it = vct.entrySet().iterator();
		while(it.hasNext()){
			Map.Entry<Integer, Double> pairs = (Map.Entry<Integer, Double>)it.next();
			double fv = pairs.getValue();
			spVct[i] = new _SparseFeature(pairs.getKey(), fv);
			i++;
		}
		Arrays.sort(spVct);		
		return spVct;
	}
	
	static public _SparseFeature[] MergeSpVcts(ArrayList<_SparseFeature[]> vcts) {
		HashMap<Integer, Double> vct = new HashMap<Integer, Double>();
		
		for(_SparseFeature[] fv:vcts) {
			for(_SparseFeature f:fv) {
				int x = f.getIndex();
				if (vct.containsKey(x)) {
					vct.put(x, vct.get(x) + f.getValue());
				} else {
					vct.put(x, f.getValue());
				}
			}
		}
		return Utils.createSpVct(vct);
	}
	
	static public _SparseFeature[] createSpVct(ArrayList<HashMap<Integer, Double>> vcts) {
		HashMap<Integer, _SparseFeature> spVcts = new HashMap<Integer, _SparseFeature>();
		HashMap<Integer, Double> vPtr;
		_SparseFeature spV;
		
		int dim = vcts.size();
		for(int i=0; i<dim; i++) {
			vPtr = vcts.get(i);
			if (vPtr==null || vPtr.isEmpty())
				continue; // it is possible that we are missing this dimension
			
			//iterate through all the features in this section
			Iterator<Entry<Integer, Double>> it = vPtr.entrySet().iterator();
			while(it.hasNext()){
				Map.Entry<Integer, Double> pairs = (Map.Entry<Integer, Double>)it.next();
				int index = pairs.getKey();
				double value = pairs.getValue();
				if (spVcts.containsKey(index)) {
					spV = spVcts.get(index);
					spV.addValue(value); // increase the total value
				} else {
					spV = new _SparseFeature(index, value, dim);
					spVcts.put(index, spV);
				}
				spV.setValue4Dim(value, i);
			}
		}
		
		int size = spVcts.size();
		_SparseFeature[] resultVct = spVcts.values().toArray(new _SparseFeature[size]);
		
		Arrays.sort(resultVct);		
		return resultVct;
	}
	
	public static String cleanHTML(String content) {
		if (content.indexOf("<!--")==-1 || content.indexOf("-->")==-1)
			return content;//clean text
		
		int start = 0, end = content.indexOf("<!--");
		StringBuffer buffer = new StringBuffer(content.length());
		while(end!=-1) {
			if (end>start)
				buffer.append(content.substring(start, end).trim());
			start = content.indexOf("-->", end) + 3;
			end = content.indexOf("<!--", start);
		}
		
		if (start<content.length())
			buffer.append(content.substring(start));
		
		return cleanVideoReview(buffer.toString());
	}
	
	public static void mergeVectors(HashMap<Integer, Double> src, HashMap<Integer, Double> dst) {
		Iterator<Entry<Integer, Double>> it = src.entrySet().iterator();
		while (it.hasNext()) {
			Map.Entry<Integer, Double> pairs = (Map.Entry<Integer, Double>)it.next();
			int index = pairs.getKey();
			if (dst.containsKey(index)==false) 
				dst.put(index, pairs.getValue());
			else
				dst.put(index, pairs.getValue() + dst.get(index));
		}
	}
	
	public static String cleanVideoReview(String content) {
		if (!content.contains("// <![CDATA[") || !content.contains("Length::"))
			return content;
		
		int start = content.indexOf("// <![CDATA["), end = content.indexOf("Length::", start);
		end = content.indexOf("Mins", end) + 4;
		StringBuffer buffer = new StringBuffer(content.length());
		buffer.append(content.substring(0, start));
		buffer.append(content.substring(end));
		
		if (buffer.length()==0)
			return null;
		else
			return buffer.toString();
	}
		
	public static boolean endWithPunct(String stn) {
		char lastChar = stn.charAt(stn.length()-1);
		return !((lastChar>='a' && lastChar<='z') 
				|| (lastChar>='A' && lastChar<='Z') 
				|| (lastChar>='0' && lastChar<='9'));
	}
	
	public static _SparseFeature[] negSpVct(_SparseFeature[] fv) {
		_SparseFeature[] result = new _SparseFeature[fv.length];
		for(int i=0; i<fv.length; i++)
			result[i] = new _SparseFeature(fv[i].getIndex(), -fv[i].getValue());
		return result;
	}
	
	//x_i - x_j 
	public static _SparseFeature[] diffVector(_SparseFeature[] spVcti, _SparseFeature[] spVctj){
		//first deal with special case
		if (spVcti==null && spVctj==null)
			return null;
		else if (spVctj==null)
			return spVcti;
		else if (spVcti==null)
			return negSpVct(spVctj);		
		
		ArrayList<_SparseFeature> vectorList = new ArrayList<_SparseFeature>();
		int i = 0, j = 0;
		_SparseFeature fi = spVcti[i], fj = spVctj[j];
		
		double fv;
		while (i < spVcti.length && j < spVctj.length) {
			fi = spVcti[i];
			fj = spVctj[j];
			
			if (fi.getIndex() == fj.getIndex()) {
				fv = fi.getValue() - fj.getValue();
				if (Math.abs(fv)>Double.MIN_VALUE)//otherwise it is too small
					vectorList.add(new _SparseFeature(fi.getIndex(),fv));
				i++; 
				j++; 
			} else if (fi.getIndex() > fj.getIndex()){
				vectorList.add(new _SparseFeature(fj.getIndex(), -fj.getValue()));
				j++;
			}
			else{
				vectorList.add(new _SparseFeature(fi.getIndex(), fi.getValue()));
				i++;
			}
		}
		
		while (i < spVcti.length) {
			fi = spVcti[i];
			vectorList.add(new _SparseFeature(fi.getIndex(), fi.getValue()));
			i++;
		}
		
		while (j < spVctj.length) {
			fj = spVctj[j];
			vectorList.add(new _SparseFeature(fj.getIndex(), -fj.getValue()));
			j++;
		}
		
		return vectorList.toArray(new _SparseFeature[vectorList.size()]);
	}
	
	//Get projectSpVct by building a map filter, added by Hongning.
	static public _SparseFeature[] projectSpVct(_SparseFeature[] fv, Map<Integer, Integer> filter) {
		ArrayList<_SparseFeature> pFv = new ArrayList<_SparseFeature>();
		for(_SparseFeature f:fv) {
			if (filter.containsKey(f.getIndex())) {
				pFv.add(new _SparseFeature(filter.get(f.getIndex()), f.getValue()));
			}
		}
		
		if (pFv.isEmpty())
			return null;
		else
			return pFv.toArray(new _SparseFeature[pFv.size()]);
	}
	
	//Get projectSpVct by building a hashmap<Integer, String> filter, added by Lin.
	static public _SparseFeature[] projectSpVct(_SparseFeature[] fv, HashMap<Integer, String> filter) {
		ArrayList<_SparseFeature> pFv = new ArrayList<_SparseFeature>();
		for(_SparseFeature f:fv) {
			if (filter.containsKey(f.getIndex())) {
				pFv.add(new _SparseFeature(f.getIndex(), f.getValue()));
			}
		}
		if (pFv.isEmpty())
			return null;
		else
			return pFv.toArray(new _SparseFeature[pFv.size()]);
	}
	
	//Sgn function: >= 0 1; < 0; 0.
	public static int sgn(double a){
		if (a >= 0) return 1;
		else return 0;
	}

}
