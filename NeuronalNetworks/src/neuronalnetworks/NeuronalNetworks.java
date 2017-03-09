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

	static int seuil = 128;
	Layer[] layers;
	public double[][][] weights;
	private double[][][] weights2;
	private int numberOfLayers;
	
	//Conversion de l'image en tableau
	public static double[] imageLecture(String location){
		BufferedImage image;
		try {
			//Lecture de l'image par JAVA
			image = ImageIO.read(new File(location));
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
	
	public void neuronalNetwork(double[][][] weights){
		this.weights = weights;
	}
	
	public String forwardPropagation() throws IOException, ClassNotFoundException{
		
		String location = "/home/timoth/Documents/TSP/1A/Projet Info/images/";
		//TODO Modify images location
				
		String imageName = "0_082";
		//TODO Modify image name
		
		double[] image = imageLecture(location + imageName +".png");
		this.numberOfLayers = weights.length;
		
		//Creation du tableau de couches
		layers = new Layer[numberOfLayers+1];
		layers[0] = new Layer(image, weights[0]);
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
		
		layers[0].execute();
		
		String result="";
		
		return result;
	}
	
	public void backPropagation(String filelocation, int[] expectedResult) throws  IOException, ClassNotFoundException{
		
		this.forwardPropagation();
		this.layers[this.numberOfLayers-1].backprop_init(expectedResult);
		
	}
	
	
	public void extractWeights(){
		File f = new File("Weights");
		//Extraction de l'objet weights
		if(f.exists()){
			FileInputStream fis;
			try {
				fis = new FileInputStream ("Weights");
				ObjectInputStream ois = new ObjectInputStream (fis);
				Object weight = ois.readObject();
				weights2 = (double[][][]) weight;
				weights = weights2;
				ois.close();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ClassNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	public void saveWeights(){
		//Enregistrement de l'objet weights
		FileOutputStream fos;
		
		try {
			fos = new FileOutputStream ("Weights");
			ObjectOutputStream oos = new ObjectOutputStream (fos);
			oos.writeObject(this.weights);
			oos.close();
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
	}
	
	public int Bernoulli(double p){
		double x = Math.random();
		if(x<p){
			return 1;
		}
		else{
			return -1;
		}
	}
}
