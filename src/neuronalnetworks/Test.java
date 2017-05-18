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
	
	public static void loadImages(){
		for(int i=0; i<10; i++){
			for(int j=1; j<4000; j++){
				String nom = i + "_0" + j + ".png";
				nom = NeuronalNetworks.location + "\\projetinfo\\images\\" + nom;
				images[i][j]=NeuronalNetworks.imageLecture(nom);
			}
		}
		centreReduitImages();
		System.out.println("Les images ont ete charg√©s en ram");
	}
	
	// La variance et la moyenne sont calculÈes pour chaque pixel et non pas globalement
	public static double[] average(){
		double[] avg = new double[784];
		for(int i=0; i<10; i++){
			for(int j=0; j<images[i].length; j++){
					arraySum(avg,images[i][j]);
			}
		}
		return avg;
	}
	
	private static double[] arraySum(double[] a1, double[] a2) {
		
		assert (a1.length == a2.length);
		
		double[] resultat = new double[a1.length];
		
		for (int i = 0 ; i < a1.length ; i++) {
			
			resultat[i] += a1[i];
			resultat[i] += a2[i];
			
		}
		
		return resultat;
		
		
	}
	
	public static double[] variance(double[] moyenne) {
		
		double[] var = new double[784];
		for (int i = 0 ; i < 10 ; i++) {
			for (int j = 0; j < images[i].length ; j++) {
					
					arraySum(var, arraySquared(images[i][j]));
					
			}
		}
		
		return arraySum(var, arrayNegate(arraySquared(moyenne)));
		
		
	}
	
	private static double[] arrayNegate(double[] a) {
		
		double[] resultat = new double[a.length];
		for (int i = 0 ; i < a.length ; i++)
			resultat[i] -= a[i];
		return resultat;
		
	}
	
	private static double[] arraySquared(double[] a) {
		
		double[] resultat = new double[a.length];
		
		for (int i = 0 ; i < a.length ; i++) {
			
			resultat[i] = Math.pow(a[i], 2);
			
		}
		
		return resultat;
		
		
	}
	
	private static double[] arraySqrt(double[] a) {
		
		double[] resultat = new double[a.length];
		for(int i = 0 ; i < a.length ; i++)
			resultat[i] = Math.sqrt(a[i]);
		return resultat;
		
	}
	
	// /!\ Il faut rajouter -ea dans les paramËtres de la vm (clic droit > Run As > Run Configurations)
	
	public static void centreReduitImages(){
		double[] moyenne = average();
		double[] var = variance(moyenne);
		double[] ecartType = arraySqrt(var);
		for(int i=0; i<10; i++){
			for(int j=0; j<images[i].length; j++){
				for(int k=0; k<784; k++){
					assert (Math.abs(ecartType[k]) >= 0.000001);
					assert (!(Double.isNaN(images[i][j][k])));
					assert (!(Double.isNaN(moyenne[k])));
					assert (Double.isFinite(images[i][j][k]));
					assert (Double.isFinite(moyenne[k]));
					images[i][j][k] = (images[i][j][k]-moyenne[k])/ecartType[k];
					assert (-1 <= images[i][j][k] && images[i][j][k] <= 1);
					assert (!(Double.isNaN(images[i][j][k])));
					System.out.println(images[i][j][k]);
				}
			}
		}
		/*
		double nouvelleMoyenne = average();
		double nouvelleVariance = variance(nouvelleMoyenne);
		assert (Math.abs(nouvelleMoyenne) <= 1);
		assert (nouvelleVariance <= 2);
		*/
		
	}
	
	public Test(int i){
		N = new NeuronalNetworks(i,false);				
	}
	
	public void extractNeuralNetworks(){
		bestNeuralNetworks.extractWeights(482, true);
		bestNeuralNetworks.extractSuccessRate();
		bestNeuralNetworks.extractLearningFactor();
		bestNeuralNetworks.extractMeanSquareError();
		System.out.println("Le r√©seau de neurones anciennnement connu a ete charge");
	}
	
	public static void saveNeuralNetworks(){
		//Enregistrement des objets necessaires pour reconstituer le reseau de neurones avec le meilleur taux de succÔøΩs
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
	
	public void learningRAM(int learningFactor){
		int count = 0;
		double moyenne = 0;
		for (int j=0; j<10; j++){
			for (int i=0; i<2000; i++){
				count++;
				try {
					moyenne+=N.backPropagationRAM(images[j][i],j, (int) (learningFactor/Math.log(count+10)));
				} catch (ClassNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		moyenne /= 20000;
		assert (moyenne <= 10);
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
				System.out.println("Taille couche cach√©e : "+ i);
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
