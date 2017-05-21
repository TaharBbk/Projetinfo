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
	Layer[] layers;
	public double[][] images = new double[60000][784];
	public double[][][] weights;
	private double[][][] weights2;
	private int numberOfWeights;
	public static double LEARNING_FACTOR;
	double successRate;
	public static double MeanSquareError;
	public static final int nombreIterationsBackprop = 1;
	
	//Conversion de l'image en tableau
	public static double[] imageLecture(String locationImage){
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

	public NeuronalNetworks(int l, boolean extract) {
		//Extraction des poids
		this.extractWeights(l, extract);		
		this.numberOfWeights = weights.length;
		//Creation du tableau de couches
		layers = new Layer[numberOfWeights+1];
		layers[numberOfWeights] = new Layer(new double[10]);

		//Creation des objets couches
		for(int i=numberOfWeights-1; i>=0; i--){
			layers[i] = new Layer(new double[weights[i][0].length], weights[i], layers[i+1]);		
		}
	}
	
	public double[] forwardPropagationRAM(double [] image) throws IOException, ClassNotFoundException{
		
		assert (image.length == 784);
		
		layers[0].setValues(image);
		layers[0].forward_init();
		
		return layers[2].getValues();
	}
	
	//return the maximum of a table
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

	public void backPropagationRAM(double[] image, int expectedResult, double learningFactor) throws  IOException, ClassNotFoundException{
		
		assert (image.length == 784);
		
		int[] expected = new int[10];
		Arrays.fill(expected, -1);
		expected[expectedResult] = 1;	
		
		//double moyenne = avgArray(this.forwardPropagationRAM(image));
		
		this.forwardPropagationRAM(image);
		
		for (int i = 0; i < nombreIterationsBackprop ; i ++) {
			
			this.layers[2].backprop_init(expected, learningFactor);
			
		}

		//return moyenne;
	}
	
	public double avgArray(double[] Tableau) {
		
		double resultat = 0;
		for (int i = 0 ; i < Tableau.length ; i++) {
			
			resultat += Tableau[i];
			
		}
		
		return resultat;
		
	}
	
	public double randomWeights(double nombreEntrees){
		double cte = Math.cbrt(3/(2*nombreEntrees));
		double x = (Math.random()*2*cte) - cte;
		while (Math.abs(x) < 0.0001)
			x = (Math.random()*2*cte) - cte;
		return x;
	}
	
	public void generateWeights(){

		for(int i=0; i<weights.length; i++){
			for(int j=0; j<weights[i].length; j++){
				for(int h=0; h<weights[i][j].length; h++){
					weights[i][j][h] = randomWeights(weights.length);
				}
			}
		}
	}
	
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
	
	public void extractSuccessRate(){
		this.successRate = 0.1;
		
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

	public String getSuccessRate() {
		return "" + successRate;
	}
}
