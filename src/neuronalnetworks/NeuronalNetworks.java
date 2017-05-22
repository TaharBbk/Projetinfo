package neuronalnetworks;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;

import javax.imageio.ImageIO;

public class NeuronalNetworks {

	public final static String location = new File("").getAbsolutePath();
	static int seuil = 150;
	/**
	 * Array qui contient les couches du reseau de neurones : 0 - entree, 1 - cachee, 2 - sortie
	 */
	Layer[] layers;
	public double[][] images = new double[60000][784];
	public double[][][] weights;
	private double[][][] weights2;
	private int numberOfWeights;
	public static double LEARNING_FACTOR;
	double successRate;
	public static double MeanSquareError;
	public static final int nombreIterationsBackprop = 1;
	
	/**
	 * Charge une image sous forme d'array
	 * @param locationImage la position de l'image sur le disque
	 * @return un array qui contient les pixels de l'image
	 */
	public static double[] imageLecture(String locationImage){
		/**
		 * Objet qui permet de charger l'image depuis une position sur le disque
		 */
		BufferedImage image;
		try {
			//Lecture de l'image par JAVA
			image = ImageIO.read(new File(locationImage));
			int hauteur = image.getHeight();
			int largeur = image.getWidth();
			double[] imagetab = new double[hauteur*largeur];
			
			for (int i=0; i<hauteur; i++){
				for(int j=0; j<largeur; j++){
					Color color = new Color(image.getRGB(i,j), false);
					int couleur = (color.getBlue()+color.getRed()+color.getGreen())/3;
					if(couleur<seuil){
						imagetab[i*largeur+j]=1;
					}
					else{
						imagetab[i*largeur+j]=0;
					}
				}
			}
			return imagetab;
		}
		catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	/**
	 * Constructeur de la classe
	 * @param l taille de la couche cachee
	 * @param extract boolean qui dicte si les poids sont a extraire d'un reseau sauvegarde ou pas
	 */
	public NeuronalNetworks(int l, boolean extract) {
		//Extraction des poids
		this.extractWeights(l, extract);
		this.numberOfWeights = weights.length;
		
		
		this.layers = new Layer[numberOfWeights+1];
		this.layers[numberOfWeights] = new Layer(new double[10]);

		//Creation des objets couches
		for(int i=numberOfWeights-1; i>=0; i--){
			layers[i] = new Layer(new double[weights[i][0].length], weights[i], layers[i+1]);		
		}
	}
	
	/**
	 * Methode qui effectue une forward propagation en utilisant une image chargee en RAM
	 * @param image l'array qui contient les pixels de l'image a analyser
	 * @return l'array qui contient les valeurs de sortie du reseau de neurones apres la forward propagation
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public double[] forwardPropagationRAM(double [] image) throws IOException, ClassNotFoundException{
		
		//On verifie que l'array passe en argument est de la bonne taille
		assert (image.length == 784);
		
		layers[0].setValues(image);
		layers[0].forward_init();
		
		return layers[this.numberOfWeights].getValues();
	}
	
	/**
	 * Methode qui cherche le rang du maximum d'un array
	 * @param T array dont on cherche le rang de l'element de valeur maximum
	 * @return rang de l'element de valeur maximum de l'array
	 */
	public static int max(double[] T){
		
		double max = T[0];
		int posmax = 0;
		
		for(int i=0; i<T.length; i++){
			if(max < T[i]){
				max = T[i];
				posmax = i;
			}
		}
		
		return posmax;
	
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
			
			this.layers[2].backprop_init(expected, learningFactor);
			
		}
	}
	
	/**
	 * Methode qui renvoie la somme des elements d'un array
	 * @param Tableau array dont on cherche a sommer les elements
	 * @return somme des elements de l'array passe en argument
	 */
	public double avgArray(double[] Tableau) {
		
		double resultat = 0;
		for (int i = 0 ; i < Tableau.length ; i++) {
			
			resultat += Tableau[i];
			
		}
		
		return resultat;
		
	}
	
	/**
	 * Fonction qui genere un nombre aleatoire de loi uniforme de moyenne 0 et de variance determinee par l'argument
	 * @param nombreEntrees un coefficient qui determine la variance de la fonction, qui est de 1/nombreEntrees
	 * @return un nombre aleatoire qui depend des arguments
	 */
	public double randomWeights(double nombreEntrees){
		
		double cte = Math.cbrt(3/(2*nombreEntrees));
		double x = (Math.random()*2*cte) - cte;
		
		// On evite que le nombre soit trop faible pour ne pas avoir de poids inutile dans le reseau de neurones
		while (Math.abs(x) < 0.0001)
			x = (Math.random()*2*cte) - cte;
		return x;
	}
	
	/**
	 * Methode qui pour chaque poids assigne une valeur aleatoire qui depend de la largeur de la matrice de poids dans laquelle il se trouve
	 */
	public void generateWeights(){

		for(int i=0; i<weights.length; i++){
			for(int j=0; j<weights[i].length; j++){
				for(int h=0; h<weights[i][j].length; h++){
					weights[i][j][h] = randomWeights(weights.length);
				}
			}
		}
	}
	
	/**
	 * Methode qui charge les matrices de poids d'un reseau de neurones sauvegarde ou genere de nouveaux poids aleatoires
	 * @param i Taille de la couche cachee dans le reseau a charger / generer
	 * @param extract boolean qui indique si oui ou non il faut charger des poids d'un reseau de neurones sauvegarde, ou bien en generer de nouveaux
	 */
	public void extractWeights(int i, boolean extract){
		double[][][] weights = new double[2][][];
		weights[0] = new double[784][i];
		weights[1] = new double[i][10];
		this.weights = weights;
		
		File f = new File(location + "/bestWeights");
		//Extraction de l'objet weights
		if(f.exists() && extract){
			FileInputStream fis;
			try {
				fis = new FileInputStream (location + "/bestWeights");
				ObjectInputStream ois = new ObjectInputStream (fis);
				Object weight = ois.readObject();
				weights2 = (double[][][]) weight;
				weights = weights2;
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
	 */
	public void extractSuccessRate(){
		this.successRate = 0;
		
		File f = new File(location + "/bestSuccessRate");
		//Extraction du taux de succes
		if(f.exists()){
			FileInputStream fis;
			try {
				fis = new FileInputStream (NeuronalNetworks.location + "/bestSuccessRate");
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
	 */
	public void extractMeanSquareError(){
		NeuronalNetworks.MeanSquareError = 1;
		
		FileInputStream fis;
		//Extraction du learning factor
		File f = new File(location + "/bestMeanSquareError");
		if(f.exists()){
			try {
				fis = new FileInputStream (NeuronalNetworks.location + "/bestMeanSquareError");
				ObjectInputStream ois = new ObjectInputStream (fis);
				Object meanSquareError= ois.readObject();
				NeuronalNetworks.MeanSquareError = (double) meanSquareError;
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
	 */
	public void extractLearningFactor(){
		NeuronalNetworks.LEARNING_FACTOR = 15;
		
		FileInputStream fis;
		//Extraction du learning factor
		File f = new File(location + "/bestLearningFactor");
		if(f.exists()){
			try {
				fis = new FileInputStream (NeuronalNetworks.location + "/bestLearningFactor");
				ObjectInputStream ois = new ObjectInputStream (fis);
				Object learningFactor= ois.readObject();
				NeuronalNetworks.LEARNING_FACTOR = (int) learningFactor;
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
	 * Methode qui retourne le taux de succes de ce reseau de neurones au cours de la validation
	 * @return
	 */
	public String getSuccessRate() {
		return "" + successRate;
	}
}
