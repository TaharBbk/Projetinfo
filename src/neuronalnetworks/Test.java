package neuronalnetworks;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Arrays;

public class Test {
	public static double[][][] betterweights;
	public static double[][][] images = new double[10][4000][784];
	public static NeuronalNetworks bestNeuralNetworks;
	public static NeuronalNetworks N;
	public static double avg;
	
	public static void loadImages(){
		for(int i=0; i<10; i++){
			for(int j=1; j<4000; j++){
				String nom = i + "_0" + j + ".png";
				nom = NeuronalNetworks.location + "/images/" + nom;
				images[i][j]=NeuronalNetworks.imageLecture(nom);
			}
		}
		centreReduitImages();
		System.out.println("Les images ont ete chargés en ram");
	}
	
	public static void average(){
		double temp;
		double avg2 = 0;
		for(int i=0; i<10; i++){
			avg2 = 0;
			for(int j=0; j<images[i].length; j++){
				temp = 0;
				for(int k=0; k<784; k++){
					temp+=images[i][j][k];
				}
				avg2+=temp/784;
			}
			avg+=avg2/4000;
		}
		avg /= 10;
	}
	
	public static void centreReduitImages(){
		average();
		double var = avg - Math.pow(avg, 2);
		double ecartType = Math.sqrt(var);
		for(int i=0; i<10; i++){
			for(int j=0; j<images[i].length; j++){
				for(int k=0; k<784; k++){
					images[i][j][k] = (images[i][j][k]-avg)/ecartType;
				}
			}
		}
	}
	
	public Test(int i){
		N = new NeuronalNetworks(i,false);				
	}
	
	public void extractNeuralNetworks(){
		bestNeuralNetworks.extractWeights(482, true);
		bestNeuralNetworks.extractSuccessRate();
		bestNeuralNetworks.extractLearningFactor();
		bestNeuralNetworks.extractMeanSquareError();
		System.out.println("Le réseau de neurones anciennnement connu a ete charge");
	}
	
	public static void saveNeuralNetworks(){
		//Enregistrement des objets necessaires pour reconstituer le reseau de neurones avec le meilleur taux de succ�s
		FileOutputStream fos1;
		FileOutputStream fos2;
		FileOutputStream fos3;
		FileOutputStream fos4;
		
		try {
			fos1 = new FileOutputStream (NeuronalNetworks.location + "/bestWeights");
			fos2 = new FileOutputStream (NeuronalNetworks.location + "/bestSuccessRate");
			fos3 = new FileOutputStream (NeuronalNetworks.location + "/bestLearningFactor");
			fos4 = new FileOutputStream (NeuronalNetworks.location + "/bestMeanSquareError");
			
			ObjectOutputStream oos1 = new ObjectOutputStream (fos1);
			ObjectOutputStream oos2 = new ObjectOutputStream (fos2);
			ObjectOutputStream oos3 = new ObjectOutputStream (fos3);
			ObjectOutputStream oos4 = new ObjectOutputStream(fos4);
			
			oos1.writeObject(Test.bestNeuralNetworks.weights);
			oos2.writeObject(Test.bestNeuralNetworks.successRate);
			oos3.writeObject(NeuronalNetworks.LEARNING_FACTOR);
			oos4.writeObject(NeuronalNetworks.MeanSquareError);
			
			oos1.close();
			oos2.close();
			oos3.close();
			oos4.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}
	
	public void learningRAM(int k){
		int count = 0;
		for (int j=0; j<10; j++){
			for (int i=0; i<2000; i++){
				count++;
				try {
					N.backPropagationRAM(images[j][i],j, (int) (k/Math.log(count+10)));
				} catch (ClassNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	public double successRateCalculRAM(){
		double reussit = 0;
		double[] result;
		for (int i=2000; i<4000; i++){
			for (int j=0; j<10; j++){
				try {
					result = N.forwardPropagationRAM(images[j][i]);
					if (NeuronalNetworks.max(result) == j){
						reussit = reussit +1 ;
					}
				}
				catch (ClassNotFoundException e) {
					e.printStackTrace();
				}
				catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return reussit/20000;
	}
	
	public double rangMax(double[] input) {
		
		int rang = 0;
		double max = input[0];
		
		for (int i = 1 ; i < input.length ; i++) {
			
			if (max < input[i]) {
				
				max = input[i];
				rang = i;
				
			}
			
		}
		
		return rang;
		
	}
	
	public static double meanSquareErrorRAM() {
		double error = 0;
		double[] result;
		double[] expected;
		double[] loss;
		for (int i = 2000 ; i < 4000 ; i ++) {
			
			for (int j = 0 ; j < 10 ; j++) {
				
				expected = new double[10];
				Arrays.fill(expected, -1);
				expected[j] = 1;
				try {
					result = N.forwardPropagationRAM(images[j][i]);
					loss = Layer.lossFunction(result, expected);
					for (int k = 0; k < 10 ; k++) {
						
						error += loss[k];
						
					}
					
				} catch (ClassNotFoundException e) {
					
					e.printStackTrace();
					
				} catch (IOException e) {
					
					e.printStackTrace();
					
				}
				
			}
			
		}
		
		return error / 20000;
		
		
	}
	
	public static void findTheRightOneRAM(int k, int l, int li, int lf){
		Test.loadImages();
		Test T = new Test(1);
		bestNeuralNetworks = new NeuronalNetworks(1, true);
//		T.extractNeuralNetworks();
		double bestMeanSquareError = Test.meanSquareErrorRAM();
		for(int j = li; j<lf; j++){
			for(int i=k; i<l; i++){
				NeuronalNetworks N = new NeuronalNetworks(i, false);
				Test.N = N;
				T.learningRAM(j);
				NeuronalNetworks.MeanSquareError = meanSquareErrorRAM();
				double successRAM = T.successRateCalculRAM();
				Test.N.successRate = successRAM;
				if(successRAM > Test.bestNeuralNetworks.successRate){
					NeuronalNetworks.LEARNING_FACTOR = j;
					Test.bestNeuralNetworks = Test.N;
				}
				System.out.println("Taille couche cachée : "+ i);
				System.out.println("Learning factor : " + j);
			}
		}
		System.out.println("Erreur quadratique moyenne : " + bestMeanSquareError);
		System.out.println("Taille :" + Test.bestNeuralNetworks.weights[1][1].length);
		System.out.println("Taux de succes :" + Test.bestNeuralNetworks.successRate);
		System.out.println("Learning factor :" + NeuronalNetworks.LEARNING_FACTOR);
	}
	
	public static void tempsExecution(long i){
		System.out.print(i/3600 + " h ");
		System.out.print((i%3600)/60 + " min ");
		System.out.print((i%3600)%60 + " sec");
	}
	
	public static void main(String[] args) {
		long startTime = System.currentTimeMillis();
		Test.findTheRightOneRAM(484,489,10,11);
		Test.saveNeuralNetworks();
		System.out.println("Le meilleur reseau de neurones determine a ete sauvegarde");
		long endTime   = System.currentTimeMillis();
		long totalTime = (endTime - startTime)/1000;
		Test.tempsExecution(totalTime);
	}
}
