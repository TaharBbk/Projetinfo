package neuronalnetworks;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Arrays;

public class Test {
	public static double[][][] betterweights;
	public static double[][] images = new double[20000][784];
	public static NeuronalNetworks bestNeuralNetworks;
	public static NeuronalNetworks N;
	
	public static void loadImages(){
		int n = 0;
		for(int i=0; i<10; i++){
			for(int j=1; j<2000; j++){
				String nom = i + "_0" + j + ".png";
				nom = NeuronalNetworks.location + "/images/" + nom;
				images[n]=NeuronalNetworks.imageLecture(nom);
				n++;
			}
		}
		System.out.println("Les images ont été chargés en ram");
	}
	
	public Test(int i){
		N = new NeuronalNetworks(i,false);				
	}
	
	public void extractNeuralNetworks(){
		bestNeuralNetworks.extractWeights(482, true);
		bestNeuralNetworks.extractSuccessRate();
		bestNeuralNetworks.extractLearningFactor();
		bestNeuralNetworks.extractMeanSquareError();
		System.out.println("Le réseau de neurones anciennnement connu a été chargé");
	}
	
	public static void saveNeuralNetworks(){
		//Enregistrement des objets necessaires pour reconstituer le r�seau de neurones avec le meilleur taux de succ�s
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
	
	// fonction qui doit renvoyer en sortie un nombre compris entre 0 et 9 de manière "aléatoire"
	public int uniform(){
		double a = Math.random();
		return (int)(a*10);
	}
	
	public void learning(){
		for (int i=0; i<1000; i++){
			for (int j=0; j<10; j++){
				String nom = j + "_0" + i ;
				try {
					N.backPropagation(nom,j);
				} catch (ClassNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	public double successRateCalcul(){
		double reussit = 0;
		double[] result;
		for (int i=1000; i<2000; i++){
			for (int j=0; j<10; j++){
				String nom = j + "_0" + i ;
				try {
					result = N.forwardPropagation(nom);
					if (NeuronalNetworks.max(result) == j && result[j] == 1){
						reussit = reussit +1 ;
					}
				} catch (ClassNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return reussit/20000; 
	}
	
	public void learningRAM(int k){
		int count = 0;
		for (int j=0; j<10; j++){
			for (int i=0; i<1000; i++){
				count++;
				try {
					int id = j*2000 +i;
					N.backPropagationRAM(images[id],j, (int) (k/Math.sqrt(count)));
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
		for (int i=1000; i<2000; i++){
			for (int j=0; j<10; j++){
				try {
					int id = j*2000+i;
					result = N.forwardPropagationRAM(images[id]);
					if (NeuronalNetworks.max(result) == j && result[j]/1.7159 >= 0.9){
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
		return reussit/10000;
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
		for (int i = 1000 ; i < 2000 ; i ++) {
			
			for (int j = 0 ; j < 10 ; j++) {
				
				expected = new double[10];
				Arrays.fill(expected, -1);
				expected[j] = 1;
				try {
					int id = j*2000 + i;
					result = N.forwardPropagationRAM(images[id]);
					loss = Layer.lossFunction(result, expected);
					for (int k = 0; k < 10 ; k++) {
						
						error += loss[k];
						
					}
					
					error /= 10000;
					
				} catch (ClassNotFoundException e) {
					
					e.printStackTrace();
					
				} catch (IOException e) {
					
					e.printStackTrace();
					
				}
				
			}
			
		}
		
		return error;
		
		
	}
	
	public static double successRateRAM() {
		double success = 0;
		double[] result;
		double[] expected;
		double[] loss;
		double error;
		for (int i = 1000 ; i < 2000 ; i ++) {
			
			for (int j = 0 ; j < 10 ; j++) {
				
				expected = new double[10];
				Arrays.fill(expected, -1);
				expected[j] = 1;
				
				try {
					int id = j*2000 + i;
					error = 0;
					result = N.forwardPropagationRAM(images[id]);
					loss = Layer.lossFunction(result, expected);
					
					for (int k = 0; k < 10 ; k++) {
						
						error += loss[k];
						
					}
					
					if (error < 0.1)
						success++;
					
										
				} catch (ClassNotFoundException e) {
					
					e.printStackTrace();
					
				} catch (IOException e) {
					
					e.printStackTrace();
					
				}
				
			}
			
		}
		
		return success/10000;
		
		
	}
	
	
	public static void findTheRightOneRAM(int k, int l, int li, int lf){
		Test.loadImages();
		Test T = new Test(1);
		bestNeuralNetworks = new NeuronalNetworks(1, true);
		T.extractNeuralNetworks();
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
			}
		}
		System.out.println("Erreur quadratique moyenne : " + bestMeanSquareError);
		System.out.println("Taille :" + Test.bestNeuralNetworks.weights[1][1].length);
		System.out.println("Taux de succés :" + Test.bestNeuralNetworks.successRate);
		System.out.println("Learning factor :" + NeuronalNetworks.LEARNING_FACTOR);
	}
	
	public static void tempsExecution(long i){
		System.out.print(i/3600 + " h ");
		System.out.print((i%3600)/60 + " min ");
		System.out.print((i%3600)%60 + " sec");
	}
	
	public static void main(String[] args) {
		long startTime = System.currentTimeMillis();
		Test.findTheRightOneRAM(782,784,14,16);
		Test.saveNeuralNetworks();
		System.out.println("Le meilleur réseau de neurones déterminé a été sauvegardé");
		long endTime   = System.currentTimeMillis();
		long totalTime = (endTime - startTime)/1000;
		Test.tempsExecution(totalTime);
	}
}
