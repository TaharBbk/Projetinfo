package neuronalnetworks;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;

public class NeuralNetworks {
	//Definitions des variables
	public final static String location = new File("").getAbsolutePath(); //On cree un fichier vide pour recuperer le chemin absolu
	Layer[] layers; //Tableau contenant les differentes couches constituant le reseau
	public double[][][] weights; //Tableau contenant les matrices des poids
	private int numberOfWeights; //Nombre de matrices de poids
	public static double LEARNING_FACTOR;
	double successRate;
	public static double MeanSquareError;
	public static final int nombreIterationsBackprop = 1;

	
	//Construction du reseau de neurones
	public NeuralNetworks(int l, boolean extract) {
		//Extraction des poids
		//Si extract=true > les poids sont extraits du fichier sinon ils sont generes aleatoirement 
		this.extractWeights(l, extract);
		numberOfWeights = weights.length;
		//Creation du reseau de neurones et de ses differentes couches 
		layers = new Layer[numberOfWeights+1];
		layers[numberOfWeights] = new Layer(new double[10]);
		//Attribution des poids aux différentes couches ainsi que lien entre les différentes couches
		for(int i=numberOfWeights-1; i>=0; i--){
			layers[i] = new Layer(new double[weights[i][0].length], weights[i], layers[i+1]);		
		}
	}
	
	
	//Applique la methode de la forward propagation sur le tableau image et allume les neurones en sortie
	public double[] forwardPropagationRAM(double [] image) throws IOException, ClassNotFoundException{
		assert (image.length == 784);
		layers[0].setValues(image);
		layers[0].forward_init();
		return layers[2].getValues();
	}
	
	
	//Renvoie la position du maximum d'un tableau
	public static int max(double[] T){
		int longueur = T.length;
		double max = T[0];
		int posmax = 0;
		for(int i=0; i<longueur; i++){
			if(max < T[i]){
				max = T[i];
				posmax = i;
			}
		}
		return posmax;
	}
	
	
	//Applique la methode de la backward propagation sur le tableau image et allume les neurones en sortie (apprentissage)
	public void backPropagationRAM(double[] image, int expectedResult, double learningFactor) throws  IOException, ClassNotFoundException{
		assert (image.length == 784);
		double[] expected = new double[10];
		Arrays.fill(expected, -1);
		expected[expectedResult] = 1;	
		this.forwardPropagationRAM(image);
		for (int i = 0; i < nombreIterationsBackprop ; i ++) {			
			this.layers[2].backprop_init(expected, learningFactor);			
		}
	}
	
	
	//Permet de generer la valeur des poids pour qu'ils soient centre reduit
	public double randomWeights(double nombreEntrees){
		double cte = Math.cbrt(3/(2*nombreEntrees));
		double x = (Math.random()*2*cte) - cte;
		while (Math.abs(x) < 0.0001)
			x = (Math.random()*2*cte) - cte;
		return x;
	}
	
	
	//Genere les poids aletoirement
	public void generateWeights(){
		for(int i=0; i<weights.length; i++){
			for(int j=0; j<weights[i].length; j++){
				for(int h=0; h<weights[i][j].length; h++){
					weights[i][j][h] = randomWeights(weights.length);
				}
			}
		}
	}
	
	
	//Extrait les poids depuis le fichier si extract = true, les genere sinon
	public void extractWeights(int i, boolean extract){
		//Besoin d'initialiser les poids si le fichier n'existe pas
		double[][][] weights = new double[2][][];
		weights[0] = new double[784][i];
		weights[1] = new double[i][10];
		this.weights = weights;
		
		File f = new File(location + "/bestWeights");
		if(f.exists() && extract){
			FileInputStream fis;
			try {
				fis = new FileInputStream (location + "/bestWeights");
				ObjectInputStream ois = new ObjectInputStream (fis);
				Object weight = ois.readObject();
				weights = (double[][][]) weight;
				ois.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
			}
		}
		else {
			
			this.generateWeights();
			
		}
	}
	
	
	//Sauvegarde les poids dans un fichier
	public void saveWeights(){
		//Enregistrement de l'objet weights
		FileOutputStream fos;
		
		try {
			fos = new FileOutputStream (location + "/Weights");
			ObjectOutputStream oos = new ObjectOutputStream (fos);
			oos.writeObject(this.weights);
			oos.close();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}
	
	
	//Extraction du taux de succes
	public void extractSuccessRate(){
		//Besoin d'initialiser le taux de succes si le fichier n'existe pas
		this.successRate = 0.1;
		
		File f = new File(location + "/bestSuccessRate");
		if(f.exists()){
			FileInputStream fis;
			try {
				fis = new FileInputStream (NeuralNetworks.location + "/bestSuccessRate");
				ObjectInputStream ois = new ObjectInputStream (fis);
				Object successRate= ois.readObject();
				this.successRate = (double) successRate;
				ois.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
			}
		}
	}
	
	
	//Extraction du learning factor
	public void extractMeanSquareError(){
		//Besoin d'initialiser l'erreur quadratique moyenne si le fichier n'existe pas
		NeuralNetworks.MeanSquareError = 1;
		
		FileInputStream fis;
		File f = new File(location + "/bestMeanSquareError");
		if(f.exists()){
			try {
				fis = new FileInputStream (NeuralNetworks.location + "/bestMeanSquareError");
				ObjectInputStream ois = new ObjectInputStream (fis);
				Object meanSquareError= ois.readObject();
				NeuralNetworks.MeanSquareError = (double) meanSquareError;
				ois.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
			}
		}
	}
	
	
	//Extraction du learning factor
	public void extractLearningFactor(){
		//Besoin d'initialiser le learning factor si le fichier n'existe pas
		NeuralNetworks.LEARNING_FACTOR = 15;
		
		FileInputStream fis;
		File f = new File(location + "/bestLearningFactor");
		if(f.exists()){
			try {
				fis = new FileInputStream (NeuralNetworks.location + "/bestLearningFactor");
				ObjectInputStream ois = new ObjectInputStream (fis);
				Object learningFactor= ois.readObject();
				NeuralNetworks.LEARNING_FACTOR = (double) learningFactor;
				ois.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
			}
		}
	}

	
	//Renvoie le taux de succes
	public String getSuccessRate() {
		return "" + successRate;
	}
}
