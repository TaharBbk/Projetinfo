package neuronalnetworks;
import java.io.IOException;

public class Test {
	public static double[][][] betterweights;
	public static double[][] images = new double[60000][784];
	public static double bestSuccessRate = 0.1;
	NeuronalNetworks N;
	
	public static void loadImages(){
		int n=0;
		for(int i=0; i<10; i++){
			for(int j=1; j<2000; j++){
				String nom = i + "_0" + j + ".png";
				nom = NeuronalNetworks.location + "\\images\\" + nom;
				System.out.println(nom);
				images[n]=NeuronalNetworks.imageLecture(nom);
				n++;
			}
		}
	}
	
	public Test(int i){
		N = new NeuronalNetworks(i);
		betterweights = N.weights;
		bestSuccessRate = N.successRate;
				
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
		int count = 0;
		for (int j=0; j<10; j++){
			for (int i=0; i<1000; i++){
				int id = j*2000 + i ;
				count++;
				try {
					N.backPropagationRAM(images[id],j, (int) (NeuronalNetworks.LEARNING_FACTOR/Math.sqrt(count)));
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
		System.out.println(reussit/10000);
		return reussit/10000;
	}	
	
	
	
	
	public static void findTheRightOneRAM(){
		Test.loadImages();
		Test T = new Test(481);
		T.learningRAM();
		bestSuccessRate = T.successRateCalculRAM();
		for(int i=482; i<485; i++){
			System.out.println(i);
			NeuronalNetworks N = new NeuronalNetworks(i);
			T.N = N;
			T.learningRAM();
			double successRateRAM = T.successRateCalculRAM(); 
			if(successRateRAM >= Test.bestSuccessRate){
				Test.betterweights=T.N.weights;
				Test.bestSuccessRate=successRateRAM;
			}
			System.out.println(Test.bestSuccessRate);
		}
		System.out.println(Test.bestSuccessRate);
	}
	
	public static void main(String[] args) {
		Test.findTheRightOneRAM();
	}
}