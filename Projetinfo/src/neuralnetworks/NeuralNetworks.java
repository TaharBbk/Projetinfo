package neuralnetworks;

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
	/**
	 * Array qui contient les couches du reseau de neurones : 0 - entree, 1 - cachee, 2 - sortie
	 */
	public Layer[] layers;
	/**
	* Array qui contient les matrices de poids des différentes couches: 0 - matrice de l'input, 1 - matrice de la couche cachee
	*/
	public double[][][] weights; //Tableau contenant les matrices des poids
	/**
	* Int qui indique la taille du reseau, il suffit de faire +1 pour connaitre la taille du reseau en termes de couche
	*/
	private int numberOfWeights;
	double learningFactor = 0;
	double successRate = 0;
	double meanSquareError = 0;
	int hiddenSize;
	public static final int nombreIterationsBackprop = 1;

	
	/**
	 * Constructeur de la classe
	 * @param l taille de la couche cachee a generer
	 * @param loadFrom String indiquant l'emplacement du reseau a charger
	 */
	public NeuralNetworks(String loadFrom) {
		
		assert (loadFrom != "");
		
		//Extraction des donnees
		this.extractData(loadFrom);
		numberOfWeights = weights.length;
		//Creation du reseau de neurones et de ses differentes couches 
		layers = new Layer[numberOfWeights+1];
		layers[numberOfWeights] = new Layer(new double[10]);
		//Attribution des poids aux diff�rentes couches ainsi que lien entre les diff�rentes couches
		for(int i=numberOfWeights-1; i>=0; i--){
			layers[i] = new Layer(new double[weights[i][0].length], weights[i], layers[i+1]);		
		}
	}
	
	public NeuralNetworks(int l) {
		
		//Generation des poids
		this.extractWeights(l);
		numberOfWeights = weights.length;
		//Creation du reseau de neurones et de ses differentes couches 
		layers = new Layer[numberOfWeights+1];
		layers[numberOfWeights] = new Layer(new double[10]);
		//Attribution des poids aux diff�rentes couches ainsi que lien entre les diff�rentes couches
		for(int i=numberOfWeights-1; i>=0; i--){
			layers[i] = new Layer(new double[weights[i][0].length], weights[i], layers[i+1]);		
		}
		
		
	}
	
	public NeuralNetworks() {
		
		this.weights = new double[2][][];
		weights[0] = new double[2][5];
		weights[1] = new double[5][1];
		//weights[2] = new double[2][1];
		
		this.generateWeights();
		
		this.layers = new Layer[3];
		layers[2] = new Layer(new double[1]);
		//layers[2] = new Layer(new double[weights[2][0].length], weights[2], layers[3]);
		layers[1] = new Layer(new double[weights[1][0].length], weights[1], layers[2]);
		layers[0] = new Layer(new double[weights[0][0].length], weights[0], layers[1]);
		
	}
	
	
	/**
	 * Methode qui effectue une forward propagation en utilisant une image chargee en RAM
	 * @param image l'array qui contient les pixels de l'image a analyser
	 * @return l'array qui contient les valeurs de sortie du reseau de neurones apres la forward propagation
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public double[] forwardPropagationRAM(double [] image) throws IOException, ClassNotFoundException{
		//assert (image.length == 784);
		
		layers[0].setValues(image);
		layers[0].forward_init();
		
		return layers[layers.length-1].getValues();
		
	}
	
	
	/**
	 * Methode qui cherche le rang du maximum d'un array
	 * @param T array dont on cherche le rang de l'element de valeur maximum
	 * @return rang de l'element de valeur maximum de l'array
	 */
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
	
	
	public void backprop(double[] input, double[] expectedOutput, double learningFactor) throws IOException, ClassNotFoundException {
		
		
		
		this.forwardPropagationRAM(input);
		
		this.layers[2].backprop_start(expectedOutput, learningFactor);
		
	}
	
	/**
	 * Methode qui effectue une backprop sur une image chargee en RAM
	 * @param image l'array contenant les pixels de l'image a analyser
	 * @param expectedResult l'array contenant les resultats attendus a la sortie du reseau de neurones
	 * @param learningFactor le facteur d'apprentissage souhaite pour cette backprop
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public void backPropagationRAM(double[] image, int expectedResult, double learningFactor) throws  IOException, ClassNotFoundException{
		
		assert (image.length == 784);
		
		double[] expected = new double[10];
		Arrays.fill(expected, -1);
		expected[expectedResult] = 1;	
		
		//On effectue d'abord la forward propagation afin de calculer les valeurs prises par les neurones, necessaires pour effectuer la backprop
		this.forwardPropagationRAM(image);
		
		//On itere le processus de backprop un nombre de fois defini
		for (int i = 0; i < nombreIterationsBackprop ; i ++) {

			this.layers[2].backprop_start(expected, learningFactor);		

		}
	}
	
	
	/**
	 * Fonction qui genere un nombre aleatoire de loi uniforme de moyenne 0 et de variance determinee par l'argument
	 * @param nombreEntrees un coefficient qui determine la variance de la fonction, qui est de 1/nombreEntrees
	 * @return un nombre aleatoire qui depend des arguments
	 */
	public double randomWeights(double nombreEntrees){

		double cte = Math.cbrt(3/(2*nombreEntrees));
		//cte = 1;
		double x = (Math.random()*2*cte) - cte;

		// On evite que le nombre soit trop faible pour ne pas avoir de poids inutile dans le reseau de neurones
		while (Math.abs(x) < 0.05)

			x = (Math.random()*2*cte) - cte;
		
		return ((Math.random()*0.1)-0.05);
	}
	
	
	/**
	 * Methode qui pour chaque poids assigne une valeur aleatoire qui depend de la largeur de la matrice de poids dans laquelle il se trouve
	 */
	public void generateWeights(){
		for(int i=0; i<weights.length; i++){
			for(int j=0; j<weights[i].length; j++){
				for(int h=0; h<weights[i][j].length; h++){
					weights[i][j][h] = randomWeights(weights[0].length);
					//System.out.println(weights[i][j][h]);
					assert(weights[i][j][h] != 0);
				}
			}
		}
	}
	
	
	/**
	 * Methode qui genere de nouveaux poids aleatoires
	 * @param i Taille de la couche cachee dans le reseau a generer
	 */
	public void extractWeights(int i){
		this.weights = new double[2][][];
		weights[0] = new double[784][i];
		weights[1] = new double[i][10];
			
		this.generateWeights();
	
	}
	
	
	/**
	 * Methode qui extrait des poids a partir d'un fichier
	 * @param loadFrom String indiquant le fichier de poids dont il s'agit
	 */
	public void extractWeights(String loadFrom) {
		
		//Extraction de l'objet weights
		FileInputStream fis;
		try {
			
			fis = new FileInputStream (location + "/Weights_"+loadFrom);
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
	
	
	/**
	 * Fonction qui sauvegarde les poids de l'instance actuelle de la classe, afin de les reutiliser plus tard
	 */
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
	
	
	/**
	 * Fonction qui permet de charger le taux de succes sauvegarde
	 * @param loadFrom String indiquant le fichier concerne
	 */
	public void extractSuccessRate(String loadFrom){
		//Besoin d'initialiser le taux de succes si le fichier n'existe pas
		this.successRate = 0.1;
		
		File f = new File(location + "/SuccessRate_"+loadFrom);
		if(f.exists()){
			FileInputStream fis;
			try {
				fis = new FileInputStream (NeuralNetworks.location + "/SuccessRate_"+loadFrom);
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
	
	
	/**
	 * Fonction qui permet de charger l'erreur quadratique moyenne sauvegardee
	 * @param loadFrom String indiquant le fichier concerne
	 */
	public void extractMeanSquareError(String loadFrom){
		//Besoin d'initialiser l'erreur quadratique moyenne si le fichier n'existe pas
		this.meanSquareError = 1;
		
		FileInputStream fis;
		File f = new File(location + "/MeanSquareError_"+loadFrom);
		if(f.exists()){
			try {
				fis = new FileInputStream (NeuralNetworks.location + "/MeanSquareError_"+loadFrom);
				ObjectInputStream ois = new ObjectInputStream (fis);
				Object meanSquareError= ois.readObject();
				this.meanSquareError = (double) meanSquareError;
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
	
	
	/**
	 * Methode qui permet de charger le facteur d'apprentissage sauvegarde
	 * @param loadFrom String indiquant le fichier concerne
	 */
	public void extractLearningFactor(String loadFrom){
		//Besoin d'initialiser le learning factor si le fichier n'existe pas
		this.learningFactor = 0.2;
		
		FileInputStream fis;
		File f = new File(location + "/LearningFactor_"+loadFrom);
		if(f.exists()){
			try {
				fis = new FileInputStream (NeuralNetworks.location + "/LearningFactor_"+loadFrom);
				ObjectInputStream ois = new ObjectInputStream (fis);
				Object learningFactor= ois.readObject();
				this.learningFactor = (double) learningFactor;
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
	
	
	/**
	* Methode qui permet de charger l'ensemble des parametres necessaire a la reconstruction du reseau
	* @param loadFrom String indiquant le fichier concerne
	*/
	public void extractData(String loadFrom) {
		
		this.extractWeights(loadFrom);
		this.extractMeanSquareError(loadFrom);
		this.extractLearningFactor(loadFrom);
		this.extractSuccessRate(loadFrom);
		
	}

	
	/**
	 * Methode qui retourne le taux de succes de ce reseau de neurones au cours de la validation
	 * @return le taux de succes du reseau
	 */
	public String getSuccessRate() {
		return "" + successRate;
	}
}
