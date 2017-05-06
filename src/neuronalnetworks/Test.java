package neuronalnetworks;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;


public class Test {
	public static double[][] images = new double[20000][784];
	public static NeuronalNetworks bestNeuralNetworks;
	NeuronalNetworks N;
	
	public Test(int i){
		N = new NeuronalNetworks(i,false);
	}
	
	public static void loadImages(){
		int n=0;
		for(int i=0; i<10; i++){
			for(int j=1; j<2000; j++){
				String nom = i + "_0" + j + ".png";
				nom = NeuronalNetworks.location + "\\images\\" + nom;
				images[n]=NeuronalNetworks.imageLecture(nom);
				n++;
			}
		}
		System.out.println("Toutes les images ont été chargé sur la ram");
	}
	
	public void extractNeuralNetworks(){
		bestNeuralNetworks.extractWeights(482, true);
		bestNeuralNetworks.extractSuccessRate();
		bestNeuralNetworks.extractLearningFactor();
		System.out.println("Le réseau de neurones anciennnement connu a été chargé");
	}
	
	public static void saveNeuralNetworks(){
		//Enregistrement des objets necessaires pour reconstituer le réseau de neurones avec le meilleur taux de succès
		FileOutputStream fos1;
		FileOutputStream fos2;
		FileOutputStream fos3;
		
		try {
			fos1 = new FileOutputStream (NeuronalNetworks.location + "/bestWeights");
			fos2 = new FileOutputStream (NeuronalNetworks.location + "/bestSuccessRate");
			fos3 = new FileOutputStream (NeuronalNetworks.location + "/bestLearningFactor");
			
			ObjectOutputStream oos1 = new ObjectOutputStream (fos1);
			ObjectOutputStream oos2 = new ObjectOutputStream (fos2);
			ObjectOutputStream oos3 = new ObjectOutputStream (fos3);
			
			oos1.writeObject(Test.bestNeuralNetworks.weights);
			oos2.writeObject(Test.bestNeuralNetworks.successRate);
			oos3.writeObject(NeuronalNetworks.LEARNING_FACTOR);
			
			oos1.close();
			oos2.close();
			oos3.close();
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
					if (result[j] == 1){
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
	
	public void learningRAM(){
		for (int j=0; j<10; j++){
			for (int i=0; i<1000; i++){
				int id = j*2000 + i ;
				try {
					N.backPropagationRAM(images[id],j);
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
				int id = j*2000 + i ;
				try {
					result = N.forwardPropagationRAM(images[id]);
					if (result[j] >= 0.9){
						reussit = reussit +1 ;
					}
				} catch (ClassNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return reussit/10000;
	}
	
	public static void findTheRightOneRAM(int k, int l){
		Test.loadImages();
		Test T = new Test(1);
		bestNeuralNetworks = new NeuronalNetworks(1,true);
		T.extractNeuralNetworks();
		for(int j=10; j<11; j++){
			for(int i=k; i<l; i++){
				NeuronalNetworks N = new NeuronalNetworks(i,false);
				T.N = N;
				NeuronalNetworks.LEARNING_FACTOR = j;
				T.learningRAM();
				double successRateRAM = T.successRateCalculRAM();
				T.N.successRate = successRateRAM;
				if(successRateRAM >= Test.bestNeuralNetworks.successRate){
					Test.bestNeuralNetworks = T.N;
				}
			}
		}
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
		Test.findTheRightOneRAM(482,484);
		Test.saveNeuralNetworks();
		System.out.println("Le meilleur réseau de neurones déterminé a été sauvegardé");
		long endTime   = System.currentTimeMillis();
		long totalTime = (endTime - startTime)/1000;
		Test.tempsExecution(totalTime);
	}
}