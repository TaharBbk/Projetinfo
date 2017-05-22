package neuralnetworks;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import javax.imageio.ImageIO;
import java.awt.Toolkit;

public class Learning {
	//Definition des variables
	static int seuil = 150; //Seuil pour les pixels des images en noir et blanc 1 si le pixel est noir et 0 sinon 
	public static double[][][] images = new double[10][5000][784]; //Conteneur des tableaux d'images en RAM
	public static NeuralNetworks bestNeuralNetworks; //Meilleur reseau de neurones connus
	public static NeuralNetworks N; //Reseau de test
	public static String OS = System.getProperty("os.name").toLowerCase(); //Permet de determiner la distribution du systeme
	
	
	//Convertit les images de la base en un tableau de 0 et de 1
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
	
	
	//Permet d'extraire les images dans un tableau qui sera charge en RAM
	public static void loadImages(){
		for(int i=0; i<10; i++){
			for(int j=1; j<5000; j++){
				String nom = i + "_0" + j + ".png";
				//Il faut determiner l'os de l'ordinateur car les chemins sont ecrit differements
				if(OS.indexOf("win") >= 0){
					nom = NeuralNetworks.location + "\\images\\" + nom;
				}
				else if(OS.indexOf("nix") >= 0 || OS.indexOf("nux") >= 0 || OS.indexOf("aix") > 0){
					nom = NeuralNetworks.location + "/images/" + nom;
				}
				images[i][j]=Learning.imageLecture(nom);
			}
		}
		centreReduitImages();
		System.out.println("Les images ont ete chargees en ram");
		System.out.println("----------------------------------");
	}
	
	
	// La variance et la moyenne sont calculees pour chaque pixel et non pas globalement
	public static double[] average(){
		double[] avg = new double[784];
		for(int i=0; i<10; i++){
			for(int j=0; j<images[0].length; j++){
					avg = arraySum(avg,images[i][j]);
			}
		}
		return arrayDivide(avg, 50000);
	}
	
	
	//Renvoie un tableau dont l'element d'indice i est la somme des elements d'indice i des tableaux passes en arguments
	private static double[] arraySum(double[] a1, double[] a2) {		
		assert (a1.length == a2.length);		
		double[] resultat = new double[a1.length];		
		for (int i = 0 ; i < a1.length ; i++) {			
			resultat[i] += a1[i];
			resultat[i] += a2[i];			
		}		
		return resultat;		
	}
	
	
	//Divise tous les elements du tableau par un diviseur
	private static double[] arrayDivide(double[] a, double diviseur) {		
		double[] resultat = new double[a.length];
		for (int i = 0 ; i < a.length ; i++) {
			resultat[i] = a[i] / diviseur;			
		}		
		return resultat;		
	}
	
	
	//Calcule la variance du tableau passe en argument
	public static double[] variance(double[] moyenne) {		
		double[] var = new double[784];
		for (int i = 0 ; i < 10 ; i++) {
			for (int j = 0; j < images[i].length ; j++) {					
					var = arraySum(var, arraySquared(images[i][j]));					
			}
		}		
		return arraySum(arrayDivide(var, 50000), arrayNegate(arraySquared(moyenne)));				
	}
	
	
	//Inverse les signes de tous les elements du tableau passe en arguments
	private static double[] arrayNegate(double[] a) {		
		double[] resultat = new double[a.length];
		for (int i = 0 ; i < a.length ; i++)
			resultat[i] -= a[i];
		return resultat;		
	}
	
	
	//Mets au carre tous les elements du tableau passe en arguments
	private static double[] arraySquared(double[] a) {		
		double[] resultat = new double[a.length];		
		for (int i = 0 ; i < a.length ; i++) {			
			resultat[i] = Math.pow(a[i], 2);			
		}		
		return resultat;		
	}
	
	
	//Appliquer la racine carre a tous les elements du tableau passe en arguments
	private static double[] arraySqrt(double[] a) {	
		double[] resultat = new double[a.length];
		for(int i = 0 ; i < a.length ; i++)
			resultat[i] = Math.sqrt(a[i]);
		return resultat;		
	}
	
	// /!\ Il faut rajouter -ea dans les parametres de la vm (clic droit > Run As > Run Configurations)
	
	
	//Permet de centrer reduire les images pour optimiser l'apprentissage
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
					assert (!(Double.isNaN(images[i][j][k])));
				}
			}
		}
	}
	
	
	//Constructeur de la classe
	public Learning(int i){
		N = new NeuralNetworks(i,false);				
	}
	
	
	//Extraction du reseaux de neurones enregistrer dans les fichiers
	public void extractNeuralNetworks(){//permet d'extraire le reseau de neurones enregistrées
		bestNeuralNetworks.extractWeights(482, true);
		bestNeuralNetworks.extractSuccessRate();
		bestNeuralNetworks.extractLearningFactor();
		bestNeuralNetworks.extractMeanSquareError();
		System.out.println("Le reseau de neurones anciennnement connu a ete charge");
	}
	
	
	//Enregistrement des objets necessaires pour reconstituer le reseau de neurones avec le meilleur taux de succes
	public static void saveNeuralNetworks(){
		FileOutputStream fos1;
		FileOutputStream fos2;
		FileOutputStream fos3;
		FileOutputStream fos4;
		
		try {
			fos1 = new FileOutputStream (NeuralNetworks.location + "/bestWeights");
			fos2 = new FileOutputStream (NeuralNetworks.location + "/bestSuccessRate");
			fos3 = new FileOutputStream (NeuralNetworks.location + "/bestLearningFactor");
			fos4 = new FileOutputStream (NeuralNetworks.location + "/bestMeanSquareError");
			
			ObjectOutputStream oos1 = new ObjectOutputStream (fos1);
			ObjectOutputStream oos2 = new ObjectOutputStream (fos2);
			ObjectOutputStream oos3 = new ObjectOutputStream (fos3);
			ObjectOutputStream oos4 = new ObjectOutputStream(fos4);
			
			oos1.writeObject(Learning.bestNeuralNetworks.weights);
			oos2.writeObject(Learning.bestNeuralNetworks.successRate);
			oos3.writeObject(Learning.bestNeuralNetworks.learningFactor);
			oos4.writeObject(Learning.bestNeuralNetworks.meanSquareError);
			
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
	
	
	//Apprentissage sur un echantillon de la base stocke en ram
	public void learningRAM(double learningFactor){
		int count = 0;
		for (int j=0; j<10; j++){
			for (int i=0; i<1500; i++){
				count++;
				try {
					N.backPropagationRAM(images[j][i],j, learningFactor/(2*Math.sqrt(count)));
				} catch (ClassNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	
	//Renvoie le taux de succes et l'erreur quadratique moyenne du reseau sur un echantillon de la base stocke en ram
	public double[] successRateCalculRAM(){
		double[] result;
		double success = 0;
		double eqm=0;
		double[] expected;
		double[] temp;
		for (int i=3000; i<5000; i++){
			for (int j=0; j<10; j++){
				try {
					result = N.forwardPropagationRAM(images[j][i]);
					//Si le neurone attendu a la valeur maximal, il s'agit d'un succés
					if (NeuralNetworks.max(result) == j){
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
	
	
	//Trouve le reseau ayant le meilleur taux de succes
	public static void findTheRightOneRAM(int hiddenSizeStart, int hiddenSizeEnd, double learnStart, double learnEnd, double learnIncrement){
		//On charge les images en ram pour accelerer le traitement
		Learning.loadImages();
		//On affiche le nombre d'iterations de la boucle pour voir le temps que cela peut prendre
		System.out.println("Nombre d'iterations : " + (int) ((hiddenSizeEnd-hiddenSizeStart)*((learnEnd-learnStart)/learnIncrement)));
		System.out.println("------------------------");
		Learning T = new Learning(hiddenSizeStart);
		bestNeuralNetworks = new NeuralNetworks(hiddenSizeStart, true);
		//On extrait le reseau de neurones des fichiers
		T.extractNeuralNetworks();
		double[] bestStats = new double[2]; // 0 = successrate, 1 = meanSquareError
		bestStats[0] = Learning.bestNeuralNetworks.successRate;
		bestStats[1] = Learning.bestNeuralNetworks.meanSquareError;
		//On affiche le meilleur reseaux de neurones connu a ce jour 
		System.out.println("Erreur quadratique moyenne : " + bestStats[1]);
		System.out.println("Taille : " + Learning.bestNeuralNetworks.weights[1].length);
		System.out.println("Taux de succes : " + bestStats[0]);
		System.out.println("Learning factor : " + Learning.bestNeuralNetworks.learningFactor);
		System.out.println("-----------------------");
		Learning.saveNeuralNetworks();
		double[] stats;
		double currentLearnF = learnStart;
		//On teste pour differentes valeurs du learning factor
		while(currentLearnF <= learnEnd) {
			//On affiche le learning factor actuel
			System.out.println("Learning factor : " + currentLearnF);
			//On calcule le taux de succes pour des reseaux de taille differente
			for(int i=hiddenSizeStart; i<hiddenSizeEnd; i++){
				System.out.println("Taille couche cachee : "+ i);
				Learning.N = new NeuralNetworks(i, false);
				//Apprentissage
				T.learningRAM(currentLearnF);
				//Test et determination du taux de succes
				stats = T.successRateCalculRAM();
				System.out.println("Taux de succes : " + stats[0]);
				if(stats[0] > bestStats[0]){
					//Mise a jour du reseaux et sauvegarde
					Learning.bestNeuralNetworks.learningFactor = currentLearnF;
					Learning.bestNeuralNetworks = Learning.N;
					Learning.bestNeuralNetworks.successRate=stats[0];
					Learning.bestNeuralNetworks.meanSquareError=stats[1];
					bestStats[0] = stats[0];
					bestStats[1] = stats[1];
					Learning.saveNeuralNetworks();
					Toolkit.getDefaultToolkit().beep();
					System.out.println("Le reseau de neurones a ete change et sauvegarde");
				}
				System.out.println("-----------------------");
			}
			currentLearnF += learnIncrement; 			
			//On affiche le meilleur reseau pour voir si il a change au cours de l'iteration sur la boucle
			System.out.println("Erreur quadratique moyenne : " + bestStats[1]);
			System.out.println("Taille : " + Learning.bestNeuralNetworks.weights[1].length);
			System.out.println("Taux de succes : " + bestStats[0]);
			System.out.println("Learning factor : " + Learning.bestNeuralNetworks.learningFactor);
			System.out.println("Le meilleur reseau de neurones determine a ete sauvegarde");
			System.out.println("---------------------------------------------------------");
		}
		//On affiche enfin le reseau obtenu a la fin de l'execution de l'ensemble du processus 
		System.out.println("Erreur quadratique moyenne : " + bestStats[1]);
		System.out.println("Taille : " + Learning.bestNeuralNetworks.weights[1].length);
		System.out.println("Taux de succes : " + bestStats[0]);
		System.out.println("Learning factor : " + Learning.bestNeuralNetworks.learningFactor);
		System.out.println("-----------------------");
	}
	
	
	//Permet d'afficher le temps en s sous la forme xx h yy min zz sec
	public static void tempsExecution(long i){
		System.out.print(i/3600 + " h ");
		System.out.print((i%3600)/60 + " min ");
		System.out.print((i%3600)%60 + " sec");
	}
	
	
	public static void main(String[] args) {
		Toolkit.getDefaultToolkit().beep();
		long startTime = System.currentTimeMillis();
		Learning.findTheRightOneRAM(477,480,0.15,0.16,0.005);
		long endTime   = System.currentTimeMillis();
		long totalTime = (endTime - startTime)/1000;
		Learning.tempsExecution(totalTime);
		Toolkit.getDefaultToolkit().beep();
	}
}
