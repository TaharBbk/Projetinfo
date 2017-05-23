package neuralnetworks;
import java.awt.Color;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Arrays;

import javax.imageio.ImageIO;

public class Learning {
	//Definition des variables
	/**
	 * Seuil pour la transformation de l'image en noir et blanc
	 */
	static int seuil = 150;
	/**
	 * tableau qui comprend l'ensemble des images utilisees pour l'entrainement du reseau de neurones, chargees en RAM
	 * entrainement[i] correspond a l'ensmble des images du chiffre i dans la base d'entrainement
	 * entrainement[i][j] correspond � l'array de pixels de la jeme image du chiffre i dans la base d'entrainement
	 */
	private static double[][][] entrainement = new double[10][2000][784];
	/**
	 * tableau qui comprend l'ensemble des images utilisees pour la validation du reseau de neurones, chargees en RAM
	 * validation[i] correspond a l'ensmble des images du chiffre i dans la base de validation
	 * validation[i][j] correspond � l'array de pixels de la jeme image du chiffre i dans la base de validation
	 */
	private static double[][][] validation = new double[10][2000][784];
	/**
	 * tableau qui comprend l'ensemble des images utilisees pour la validation du reseau de neurones, chargees en RAM
	 * test[i] correspond a l'ensmble des images du chiffre i dans la base de test
	 * test[i][j] correspond � l'array de pixels de la jeme image du chiffre i dans la base de test
	 */
	private static double[][][] test = new double[10][1000][784];
	/**
	 * Reseau de neurones ayant obtenu le meilleur taux de succ�s
	 */
	private NeuralNetworks bestNeuralNetworks; 
	/**
	 * Renseigne sur le type d'OS
	 */
	public static String OS = System.getProperty("os.name").toLowerCase(); 
	
	private double[] average = null;
	private double[] sigma = null;
	
	
	/**
	 * Methode qui convertit une image donnee en array de pixels 0 ou 1
	 * @param locationImage Le chemin de l'image a traiter
	 * @return Un Array de pixels 0 ou 1 correspondant � une image
	 */
	public static double[] imageLecture(String locationImage){ 
		
		/**
		 * Objet qui permet de charger l'image
		 */
		BufferedImage image;
		
		try {
			
			//Lecture de l'image par JAVA
			image = ImageIO.read(new File(locationImage));
			
			int hauteur = image.getHeight();
			int largeur = image.getWidth();
			/**
			 * Entier qui stocke � chaque it�ration la somme des troix couleurs du pixel parcouru
			 */
			int couleur;
			/**
			 * Objet qui permet de stocker les valeurs RBG d'un pixel
			 */
			Color color;
			
			double[] imagetab = new double[hauteur*largeur];			
			
			for (int i=0; i<hauteur; i++){
			
				for(int j=0; j<largeur; j++){
				
					color = new Color(image.getRGB(i,j), false);
					couleur = (color.getBlue()+color.getRed()+color.getGreen())/3;
					
					//Ici on compare la somme des trois couleurs au seuil : si c'est inf�rieur au seuil, plus proche du noir on place un 1, sinon c'est un 0;
					if(couleur<seuil)
						imagetab[i*largeur+j]=1;
					
					else
						imagetab[i*largeur+j]=0;
				
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
	 * Methode qui permet de charger les images des differentes bases en memoire RAM, pour eviter de les recharger a chaque fois qu'elles sont utilisees
	 */
	public void loadImages()
	throws Exception{
		
		/**
		 * Array qui contient la liste des interfaces des tranches de valeurs parcourues par le chargement d'image pour chaque base
		 * limits[0] - entrainement/validation
		 * limits[1] - validation/test
		 * limits[2] - test 
		 */
		int[] limits = new int[] {2000,2000,1000};
		
		/**
		 * Chaine de caract�res recevant le chemin vers le dossier ou se situent les images
		 */
		String path = NeuralNetworks.location;
		
		//Il faut determiner l'os de l'ordinateur car les chemins sont ecrit differements
		if(OS.indexOf("win") >= 0)
			path += "\\images\\";
		
		else 
			path += "/images/";
		
		//chargement de la base d'entrainement
		for(int i=0; i<10; i++){
			
			for(int j=0; j < 2000; j++){
				
				String nom = i + "_0" + (j+1) + ".png";

				entrainement[i][j]=Learning.imageLecture(path+nom);
			}
		}

		//chargement de la base de validation
		
		for(int i=0; i<10; i++){
			
			for(int j=0; j<limits[1]; j++){
				
				String nom = i + "_0" + (j+limits[0]+1) + ".png";

				validation[i][j]=Learning.imageLecture(path+nom);
			}
		}
		
		//chargement de la base de test
		for(int i=0; i<10; i++){
			
			for(int j=0; j<limits[2]; j++){
				
				String nom = i + "_0" + (1+j+limits[0]+limits[1]) + ".png";

				test[i][j]=Learning.imageLecture(path+nom);
			}
		}
		
		//On centre et reduit les images des bases pour une convergence plus rapide du reseau de neurones
		this.centreReduitImages();
		
		
		System.out.println("Les images ont ete chargees en ram");
		System.out.println("----------------------------------");
	}
	
	
	/**
	 * Methode qui calcule la valeur moyenne de chaque pixel calculee sur l'ensemble d'une base
	 * @param base La base sur laquelle seront effectuee les moyennes
	 * @return Un array qui contient les valeurs moyennes de chacun de 784 pixels calculees sur les images de la base
	 */
	public static double[] average(double[][][] base){
		
		double[] avg = new double[784];
		
		for(int i=0; i<10; i++){
			
			for(int j=0; j<base[0].length; j++){
					
				avg = arraySum(avg,base[i][j]);
			
			}
		
		}
		
		return arrayDivide(avg, 20000);
	
	}
	
	
	/**
	 * Methode qui effectue la somme de deux arrays de meme longueur
	 * @param a1 un array passe en argument
	 * @param a2 un array passe en argument
	 * @return un array pour lequel chaque indice a pour valeur la somme des deux valeurs a cet indice des arrays passes en argument
	 */
	private static double[] arraySum(double[] a1, double[] a2) {		
		assert (a1.length == a2.length);		
		double[] resultat = new double[a1.length];		
		for (int i = 0 ; i < a1.length ; i++) {			
			resultat[i] += a1[i];
			resultat[i] += a2[i];			
		}		
		return resultat;		
	}
	
	
	/**
	 * Divise tous les elements d'un array par un scalaire
	 * @param a array pass en argument
	 * @param diviseur
	 * @return array dont tous les elements ont ete divises par le diviseur
	 */
	private static double[] arrayDivide(double[] a, double diviseur) {		
		double[] resultat = new double[a.length];
		for (int i = 0 ; i < a.length ; i++) {
			resultat[i] = a[i] / diviseur;			
		}		
		return resultat;		
	}
	
	
		//Calcule la variance du tableau passe en argument
		public static double[] variance(double[][][] base, double[] moyenne) {		
		double[] var = new double[784];
		for (int i = 0 ; i < 10 ; i++) {
			for (int j = 0; j < base[i].length ; j++) {					
					var = arraySum(var, arraySquared(base[i][j]));					
			}
		}		
		return arraySum(arrayDivide(var, 20000), arrayNegate(arraySquared(moyenne)));				
	}
	
	
	/**
	 * Oppose la valeur de chaque element d'un array
	 * @param a 
	 * @return array dont tous les elements on pris leur valeur opposee
	 */
	private static double[] arrayNegate(double[] a) {		
		double[] resultat = new double[a.length];
		for (int i = 0 ; i < a.length ; i++)
			resultat[i] -= a[i];
		return resultat;		
	}
	
	
	/**
	 * Fonction qui met au carre tous les elements d'un array
	 * @param a
	 * @return
	 */
	private static double[] arraySquared(double[] a) {		
		double[] resultat = new double[a.length];		
		for (int i = 0 ; i < a.length ; i++) {			
			resultat[i] = Math.pow(a[i], 2);			
		}		
		return resultat;		
	}
	
	
	/**
	 * Fonction qui applique la fonction racine carree a tous les elements d'un array
	 * @param a
	 * @return
	 */
	private static double[] arraySqrt(double[] a) {	
		double[] resultat = new double[a.length];
		for(int i = 0 ; i < a.length ; i++)
			resultat[i] = Math.sqrt(a[i]);
		return resultat;		
	}

	
	/**
	 * Centre et reduit l'ensemble des images d'une base
	 * @param base base passee en argument
	 */
	public void centreReduitImages(){
		
		double[] moyenne = average(entrainement);
		moyenne = arraySum(moyenne, average(validation));
		moyenne = arraySum(moyenne, average(test));
		average = arrayDivide(moyenne, 3);
		
		double[] var = arraySum(moyenne, arrayNegate(arraySquared(moyenne)));
		
		for (int i = 0 ; i < var.length ; i++) {
			
			// Si l'ecart-type est trop faible on obtient des valeurs incoherentes car on reduit en divisant par l'ecart-type. On minore donc par une valeur sure
			if (var[i] < 0.0012755102){
				var[i] = (0.0012755102);
			}
			
		}
		
		sigma = arraySqrt(var);
		
		for(int i=0; i<10; i++){
		
			for(int j=0; j<1000; j++){
			
				entrainement[i][j] = centreReduit(entrainement[i][j]);
				validation[i][j] = centreReduit(validation[i][j]);
				test[i][j] = centreReduit(test[i][j]);
				
			}
			
			for (int j = 1000 ; j < 2000 ; j++) {
				
				entrainement[i][j] = centreReduit(entrainement[i][j]);
				validation[i][j] = centreReduit(validation[i][j]);				
				
			}
		}
		
	}
	
	/**
	 * Methode qui normalise une image passee en entree
	 * @param input L'array des pixels de l'image a normaliser
	 * @return L'array de l'image normalisee
	 */
	public double[] centreReduit(double[] input) {
		
		if (average == null) {
			
			try {
				this.loadImages();
			} catch (Exception e) {}
			
		}
			
		double[] result = new double[input.length];
		
		for (int i = 0 ; i < input.length ; i++)
			result[i] = (input[i]-average[i])/sigma[i];
		
		return result;
		
	}
	
	
	/**
	 * Le constructeur de la classe. Il cree simplement un reseau de neurone non entraine, de taux de succes zero afin qu'il soit remplace par le premier reseau entraine venu contre lequel il est compare.
	 */
	public Learning(){
	
		this.bestNeuralNetworks = new NeuralNetworks(490);
		this.bestNeuralNetworks.successRate = 0;				
	
	}
	
	
	/**
	 * Methode qui permet de charger un reseau de enregistre, et qui est ensuite considere comme le meilleur connu
	 * @param i Taille de la couche cachee
	 */
/*
	public void extractNeuralNetworks(String i){//permet d'extraire le reseau de neurones enregistr�es
		
		this.bestNeuralNetworks.extractWeights(i, true);
		this.bestNeuralNetworks.extractSuccessRate();
		this.bestNeuralNetworks.extractLearningFactor();
		this.bestNeuralNetworks.extractMeanSquareError();
		System.out.println("Le reseau de neurones anciennnement connu a ete charge");
	
	}
*/	
	
	/**
	 * Enregistrement du reseau de neurones et de ses caracteristiques
	 * @param N
	 * @param i
	 */
	public static void saveNeuralNetworks(NeuralNetworks N, String s){
		
		FileOutputStream fos1;
		FileOutputStream fos2;
		FileOutputStream fos3;
		FileOutputStream fos4;
		
		try {
			
			fos1 = new FileOutputStream (NeuralNetworks.location + "/Weights_" + s);
			fos2 = new FileOutputStream (NeuralNetworks.location + "/SuccessRate_" + s);
			fos3 = new FileOutputStream (NeuralNetworks.location + "/LearningFactor_" + s);
			fos4 = new FileOutputStream (NeuralNetworks.location + "/MeanSquareError_" + s);
			
			ObjectOutputStream oos1 = new ObjectOutputStream (fos1);
			ObjectOutputStream oos2 = new ObjectOutputStream (fos2);
			ObjectOutputStream oos3 = new ObjectOutputStream (fos3);
			ObjectOutputStream oos4 = new ObjectOutputStream(fos4);
			
			oos1.writeObject(N.weights);
			oos2.writeObject(N.successRate);
			oos3.writeObject(N.learningFactor);
			oos4.writeObject(N.meanSquareError);
			
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
	
	
	//Apprentissage sur un echantillon de la base stockee en ram
	public void learningRAM(double learningFactor, NeuralNetworks N){
		
		double count = 0;
		
		
		for (int i=0; i<1500; i++){
		
			for (int j=0; j<10; j++){
				
				count++;
				
				try {
				
					assert (learningFactor/(2*Math.sqrt(count)) > 0);
					N.backPropagationRAM(entrainement[j][i],j, learningFactor/(2*Math.sqrt(count)));
				
				} catch (ClassNotFoundException e) {
				
					e.printStackTrace();
				
				} catch (IOException e) {
					
					e.printStackTrace();
				
				}
			}
		}
	}

	
	//Renvoie le taux de succes et l'erreur quadratique moyenne du reseau sur un echantillon de la base stocke en ram
	public double[] successRateCalculRAM(NeuralNetworks N, double[][][] base){
		
		double[] result;
		double[] expected;
		double[] temp;
		
		double success = 0;
		double eqm=0;
		
		for (int i=0; i<base[0].length; i++){
		
			for (int j=0; j<10; j++){
			
				try {
				
					result = N.forwardPropagationRAM(base[j][i]);
					
					//Si le neurone de valeur maximale a un rang egal au chiffre qui doit etre reconnu, c'est un succes
					if (NeuralNetworks.max(result) == j){
					
						success++;
					
					}
							
					expected = new double[10];
					Arrays.fill(expected,-1);
					expected[j] = 1;
					
					temp = Layer.lossFunction(result, expected);
					
					for (int k = 0 ; k < 10 ; k++) {
				
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
		double taille = (double) base[0].length * 10;
		res[0] = success/taille;
		res[1] = eqm/taille;
		
		return res;
	}
	
	
	//Trouve le reseau ayant le meilleur taux de succes
	public void findTheRightOneRAM(int hiddenSizeStart, int hiddenSizeEnd, double learnStart, double learnEnd, double learnIncrement, String loadFrom, String saveTo){
		
		//On charge les images en ram pour accelerer le traitement
		try {
			this.loadImages();
		} catch(Exception e) {
			
		}
		
	
		File f = new File (NeuralNetworks.location + "/Weights_" + loadFrom);
		
		if (loadFrom == "") {
			
			System.out.println("Aucun reseau de neurones charge");
			saveTo = "default";
			
		}
		else if (f.exists()) {
		
			this.bestNeuralNetworks = new NeuralNetworks(loadFrom);
			this.bestNeuralNetworks.extractData(loadFrom);
			System.out.println("Le reseau de neurones " + loadFrom + " a ete charge");
			System.out.println("Ses caract�ristiques sont :");
			System.out.println("Taille : " + this.bestNeuralNetworks.weights[1].length);
			System.out.println("Learning factor : " + this.bestNeuralNetworks.learningFactor);
			System.out.println("Erreur quadratique moyenne : " + this.bestNeuralNetworks.meanSquareError);
			System.out.println("Taux de succes : " + this.bestNeuralNetworks.successRate);
			
			
		}
		else
			System.out.println("Le reseau de neurones " + loadFrom + " n'a pas ete trouve");
		
		
		/**
		 * Array qui contient � chaque it�ration les informations du meilleur r�seau de neurones test�
		 */
		double[] stats;
		
		//On teste pour differentes valeurs du learning factor
		for (double currentLearnF = learnStart ; currentLearnF <= learnEnd ; currentLearnF += learnIncrement) {
			
			//On calcule le taux de succes pour des reseaux de taille differente
			for(int i = hiddenSizeStart ; i <= hiddenSizeEnd ; i++){
			
				//On affiche le learning factor actuel
				System.out.println("Learning factor : " + currentLearnF);
				
				System.out.println("Taille couche cachee : "+ i);
				
				NeuralNetworks tested = new NeuralNetworks(i);
				
				//Apprentissage
				
				this.learningRAM(currentLearnF, tested);
				
				//Test et determination du taux de succes
				
				stats = this.successRateCalculRAM(tested, validation);
				
				System.out.println(stats[0]);
				
				tested.hiddenSize = i;
				tested.learningFactor = currentLearnF;
				tested.successRate = stats[0];
				tested.meanSquareError = stats[1];
				
				System.out.println("Taux de succes : " + stats[0]);
				
				if(stats[0] > this.bestNeuralNetworks.successRate){
				
					//Mise a jour du reseaux et sauvegarde
					Learning.saveNeuralNetworks(tested, saveTo);
					this.bestNeuralNetworks = tested;
					Toolkit.getDefaultToolkit().beep();
					System.out.println("Le reseau de neurones a ete change et sauvegarde sous le nom "+ saveTo);
				
				}
				
				System.out.println("-----------------------");
			
			}			
		
		}
		
		//On affiche enfin le reseau obtenu a la fin de l'execution de l'ensemble du processus 
		System.out.println("Meilleur r�sultat obtenu :");
		System.out.println("Erreur quadratique moyenne : " + this.bestNeuralNetworks.meanSquareError);
		System.out.println("Taille : " + this.bestNeuralNetworks.weights[1].length);
		System.out.println("Taux de succes : " + this.bestNeuralNetworks.successRate);
		System.out.println("Learning factor : " + this.bestNeuralNetworks.learningFactor);
		System.out.println("-----------------------");
		
		System.out.println("Verification a l'aide de la base de tests");
		System.out.println("Taux de succes :" + this.successRateCalculRAM(this.bestNeuralNetworks, test)[0]);
	
	}
	
	
	//Permet d'afficher le temps en s sous la forme xx h yy min zz sec
	public static void tempsExecution(long i){
		
		System.out.print(i/3600 + " h ");
		System.out.print((i%3600)/60 + " min ");
		System.out.print((i%3600)%60 + " sec");
	
	}
	
	
	public static void main(String[] args) {
		
		
		int startingHiddenSize = Integer.parseInt(args[0]);
		int endingHiddenSize = Integer.parseInt(args[1]);
		
		double startingLearnFactor = Double.parseDouble(args[2]);
		double endingLearnFactor = Double.parseDouble(args[3]);
		double learnFactorIncrement = Double.parseDouble(args[4]);
		
		String loadFrom = (args.length < 6)? "" : args[5];
		String saveTo = (args.length < 7)? "" : args[6];
		
		long startTime = System.currentTimeMillis();
		
		Learning instance = new Learning();
		
		instance.findTheRightOneRAM(startingHiddenSize,endingHiddenSize,startingLearnFactor,endingLearnFactor,learnFactorIncrement, loadFrom, saveTo);
		
		long endTime   = System.currentTimeMillis();
		long totalTime = (endTime - startTime)/1000;
		
		Learning.tempsExecution(totalTime);
	}
}
