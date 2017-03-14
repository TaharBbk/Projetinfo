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

import javax.imageio.ImageIO;

public class NeuronalNetworks {

	public String location = "/home/timoth/Documents/TSP/1A/Projet Info/";
	//TODO Modify location
	
		static int seuil = 128;
	Layer[] layers;
	public double[][][] weights;
	private double[][][] weights2;
	private int numberOfLayers;
	public static final int LEARNING_FACTOR = 1;
	
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
	
	//Programmation de la fonction cout
	public double costFunction(double[] y, double[] yprime){
		int taille=y.length;
		double sum = 0;
		
		for(int i=0; i<taille; i++){
			sum += Math.pow(y[i]-yprime[i], 2);
		}
		return sum/2;
	}
	
	public NeuronalNetworks(){
		//Extraction des poids
		this.extractWeights();
				
		this.numberOfLayers = weights.length;
		//Creation du tableau de couches
		layers = new Layer[numberOfLayers+1];
		layers[-1] = new Layer();
		
		//Creation des objets couches
		for(int i=1; i<numberOfLayers; i++){
			layers[i] = new Layer();
			layers[i].setWeights(weights[i]);
		}
		
		//Creation des liens entre les couches
		for(int i=0; i<numberOfLayers; i++){
			layers[i].next = layers[i+1];
		}
	}
	
	public int forwardPropagation(String imageId) throws IOException, ClassNotFoundException{
		
		String imageName = "images/" + imageId;
		
		double[] image = imageLecture(location + imageName +".png");
		layers[0] = new Layer(image, weights[0]);	
		layers[0].execute();
		
		double[] result = layers[numberOfLayers].getValues();
		for(int i=0; i<layers[numberOfLayers].numberOfNeurons; i++){
			if(result[i] == 1){
				return i;
			}
		}
		return -1;
	}
	
	public void backPropagation(String imageId, int[] expectedResult) throws  IOException, ClassNotFoundException{
		
		this.forwardPropagation(imageId);
		this.layers[this.numberOfLayers-1].backprop_init(expectedResult);
		this.saveWeights();
		
	}
	
	public double randomWeights(){
		double x = Math.random();
		if(x<0.5){
			return 1;
		}
		else{
			return -1;
		}
	}
	
	public void generateWeights(){
		for(int i=0; i<weights.length; i++){
			for(int j=0; j<weights[i].length; j++){
				for(int h=0; h<weights[i][j].length; h++){
					weights[i][j][h] = randomWeights();
				}
			}
		}
	}
	
	public void extractWeights(){
		File f = new File(location + "Weights");
		//TODO Modify file name
		//Extraction de l'objet weights
		if(f.exists()){
			FileInputStream fis;
			try {
				fis = new FileInputStream (location + "Weights");
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
		else{
			double[][][] weights = new double[2][][];
			weights[0] = new double[7812][15625];
			weights[1] = new double[10][7812];
			this.generateWeights();
		}
	}
	
	public void saveWeights(){
		//Enregistrement de l'objet weights
		FileOutputStream fos;
		
		try {
			fos = new FileOutputStream (location + "Weights");
			ObjectOutputStream oos = new ObjectOutputStream (fos);
			oos.writeObject(this.weights);
			oos.close();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}
}
