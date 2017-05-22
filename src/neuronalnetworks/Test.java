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
	private static String OS = System.getProperty("os.name").toLowerCase();
	
	public static void loadImages(){
		for(int i=0; i<10; i++){
			for(int j=1; j<4000; j++){
				String nom = i + "_0" + j + ".png";
				if(OS.indexOf("win") >= 0){
					nom = NeuronalNetworks.location + "\\projetinfo\\images\\" + nom;
				}
				else if(OS.indexOf("nix") >= 0 || OS.indexOf("nux") >= 0 || OS.indexOf("aix") > 0){
					nom = NeuronalNetworks.location + "/images/" + nom;
				}
				images[i][j]=NeuronalNetworks.imageLecture(nom);
			}
		}
		centreReduitImages();
		System.out.println("Les images ont ete chargees en ram");
	}
	
	// La variance et la moyenne sont calculees pour chaque pixel et non pas globalement
	public static double[] average(){
		double[] avg = new double[784];
		for(int i=0; i<10; i++){
			for(int j=0; j<images[0].length; j++){
					avg = arraySum(avg,images[i][j]);
			}
		}
		return arrayDivide(avg, 40000);
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
	
	private static double[] arrayDivide(double[] a, double diviseur) {
		
		double[] resultat = new double[a.length];
		for (int i = 0 ; i < a.length ; i++) {
			
			resultat[i] = a[i] / diviseur;
			
		}
		
		return resultat;
		
	}
	
	public static double[] variance(double[] moyenne) {
		
		double[] var = new double[784];
		for (int i = 0 ; i < 10 ; i++) {
			for (int j = 0; j < images[i].length ; j++) {
					
					var = arraySum(var, arraySquared(images[i][j]));
					
			}
		}
		
		return arraySum(arrayDivide(var, 40000), arrayNegate(arraySquared(moyenne)));
		
		
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
	
	// /!\ Il faut rajouter -ea dans les parametres de la vm (clic droit > Run As > Run Configurations)
	
	public static void centreReduitImages(){
		double[] moyenne = average();
		double[] var = variance(moyenne);
		double[] ecartType = arraySqrt(var);
		for(int i=0; i<10; i++){
			for(int j=0; j<images[i].length; j++){
				for(int k=0; k<784; k++){
					assert (!(Double.isNaN(images[i][j][k])));
					assert (!(Double.isNaN(moyenne[k])));
					assert (Double.isFinite(images[i][j][k]));
					assert (Double.isFinite(moyenne[k]));
					if (ecartType[k]<(0.0012755102)){
						ecartType[k]=(0.0012755102);
					}
					images[i][j][k] = (images[i][j][k]-moyenne[k])/ecartType[k];
					//assert (-1 <= images[i][j][k] && images[i][j][k] <= 1);
					assert (!(Double.isNaN(images[i][j][k])));
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
		System.out.println("Le réseau de neurones anciennnement connu a ete charge");
	}
	
	public static void saveNeuralNetworks(){
		//Enregistrement des objets necessaires pour reconstituer le reseau de neurones avec le meilleur taux de succes
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
	
	public void learningRAM(double learningFactor){
		//int count = 0;
		//double moyenne = 0;
		for (int j=0; j<10; j++){
			for (int i=0; i<2000; i++){
				//count++;
				try {
					N.backPropagationRAM(images[j][i],j, learningFactor);
				} catch (ClassNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		//moyenne /= 20000;
		//assert (moyenne <= 10);
	}

	public double[] successRateCalculRAM(){
		double[] result;
		double success = 0;
		double eqm=0;
		double[] expected;
		double[] temp;
		for (int i=2000; i<4000; i++){
			for (int j=0; j<10; j++){
				try {
					result = N.forwardPropagationRAM(images[j][i]);
					
					if (NeuronalNetworks.max(result) == j){
						success++;
					}
					
					expected = new double[10];
					Arrays.fill(expected,-1);
					expected[j] = 1;
					temp = Layer.lossFunction(result, expected);
					
					for (int k = 0 ; k < 10 ; k++) {
						
						assert (!(Double.isNaN(temp[k])));
						eqm += temp[k]/10;
						
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
		double[] res = new double[2];
		res[0] = success/20000;
		res[1] = eqm/20000;
		assert(!(Double.isNaN(res[1])));
		return res;
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
	
	public static void findTheRightOneRAM(int hiddenSizeStart, int hiddenSizeEnd, double learnStart, double learnEnd, double learnIncrement){
		Test.loadImages();
		Test T = new Test(hiddenSizeStart);
//		bestNeuralNetworks = new NeuronalNetworks(hiddenSizeStart, true);
//		T.extractNeuralNetworks();
		double[] bestStats = new double[2]; // 0 = successrate, 1 = meanSquareError
		double[] stats;
		double currentLearnF = learnStart;
		while(currentLearnF < learnEnd) {
			for(int i=hiddenSizeStart; i<hiddenSizeEnd; i++){
				Test.N = new NeuronalNetworks(i, false);
				T.learningRAM(currentLearnF);
				stats = T.successRateCalculRAM();
				if(stats[0] > bestStats[0]){
					NeuronalNetworks.LEARNING_FACTOR = currentLearnF;
					Test.bestNeuralNetworks = Test.N;
					bestStats[0] = stats[0];
					bestStats[1] = stats[1];
				}
				System.out.println("Taille couche cachée : "+ i);
				System.out.println("Learning factor : " + currentLearnF);
			}
			currentLearnF += learnIncrement;
		}
		System.out.println("Erreur quadratique moyenne : " + bestStats[1]);
//		System.out.println("Taille :" + Test.bestNeuralNetworks.weights[1][1].length);
		System.out.println("Taux de succes :" + bestStats[0]);
		System.out.println("Learning factor :" + NeuronalNetworks.LEARNING_FACTOR);
	}
	
	public static void tempsExecution(long i){
		System.out.print(i/3600 + " h ");
		System.out.print((i%3600)/60 + " min ");
		System.out.print((i%3600)%60 + " sec");
	}
	
	public static void main(String[] args) {
		long startTime = System.currentTimeMillis();
		Test.findTheRightOneRAM(480,481,1,1.1,0.2);
		Test.saveNeuralNetworks();
		System.out.println("Le meilleur reseau de neurones determine a ete sauvegarde");
		long endTime   = System.currentTimeMillis();
		long totalTime = (endTime - startTime)/1000;
		Test.tempsExecution(totalTime);
	}
}
