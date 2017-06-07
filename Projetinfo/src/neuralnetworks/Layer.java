package neuralnetworks;

public class Layer {

	double[] values;						// The vector of the values of the layer
	double[][] weights;						// The matrix of the weights of the layer
	Layer next = null;						// The next layer in the NN, defaulted to null
	Layer precedent = null;					// The precedent layer in the NN
	double[] differentialErrorValues;		// The derivative of the error with respect to the values of this layer
	double[][] differentialErrorWeights;	// The derivative of the error with respect to the weights of this layer
	double[] differentialErrorProduct; 		// The derivative of the error with respect to the product of the weights of the precedent layer by the values of the precedent layer
	int numberOfNeurons;					// The number of neurons in the layer
	double[] delta;
	
	/**
	 * Variable qui determine le coefficient lineaire dans la fonction d'activation
	 */
	double activationFunctionLinearCoeff = 0;
	
	
	/**
	 * Un constructeur qui permet de creer une couche en prenant une matrice de poids donnee
	 * @param val le vecteur des valeurs des neurones de la couche
	 * @param weight la matrice de poids que possede la couche
	 * @param next la couche suivante dans le reseau de neurones
	 */
	public Layer(double[] val, double[][] weight, Layer next){
		this.values = val;
		this.weights = weight;
		this.next = next;
		this.numberOfNeurons = this.values.length;
		this.next.setPrecedent(this);
	}
	
	
	/**
	 * Un constructeur qui ne demande pas de matrice de poids a la creation de la couche
	 * @param val La liste des valeurs des neurones de la couche
	 */
	public Layer(double[] val){
		this.values = val;
		this.numberOfNeurons = this.values.length;
	}
	
	/**
	 * Une methode pour obtenir les poids de la couche
	 * @return la matrice des poids
	 */
	public double[][] getWeights() {
		return weights;
	}
	
	/**
	 * Une methode pour changer les poids de la couche
	 * @param weights les nouveaux poids de la couche
	 */
	public void setWeights(double[][] weights) {
		this.weights = weights;
	}

	/**
	 * Une methode pour assigner des valeurs aux neurones de la couche
	 * @param values le vecteur des nouvelles valeurs
	 */
	public void setValues(double[] values) {
		//System.out.println(this.values[0]);
		this.values = values;
		//System.out.println(this.values[0]);
	}
	
	/**
	 * Une methode pour definir la couche precedente dans le reseau de neurones
	 * @param p la couche etant a definir comme etant la precedente de celle-ci
	 */
	public void setPrecedent(Layer p) {
		this.precedent = p;
	}
	
	/**
	 * Une methode pour obtenir la couche precedent cette couche
	 * @return la couche precedent cette couche si elle existe, null sinon
	 */
	public Layer getPrecedent() {
		return this.precedent;
	}
	
	/**
	 * Une methode pour definir la couche suivant celle-ci dans le reseau de neurones
	 * @param n la couche a definir comme etant la suivante
	 */
	public void setNext(Layer n) {
		this.next = n;
	}

	/**
	 * Une methode pour recuperer la valeur des neurones de la couche cachee
	 * @return la valeur des neurones de la couche
	 */
	public double[] getValues(){
		return this.values;
	}
	
	/**
	 * Une methode pour appliquer la fonction d'activation aux valeurs de cette couche
	 */
	public void activate(){
		//System.out.println(this.values[0]);
		this.values=activationFunction(this.values);
		//System.out.println(this.values[0]);
	}
	
	/**
	 * Une methode pour effectuer la forward propagation
	 */
	public void propagate(){
		//On verifie qu'aucune valeur qui a ete entree est NaN
		for (int i = 0 ; i < this.values.length ; i++) {
			assert(!(Double.isNaN(this.values[i])));
		}
		
		//On effectue l'activation des valeurs
		this.activate();
		
		/*S'il existe une couche suivante, on modifie ses valeurs comme etant 
		 * le resultat de la multiplication de la matrice des poids de cette couche
		 * avec les valeurs de cette couche*/
		if(this.next!=null){
			this.next.setValues(productMatrixVector(this.weights, this.values));
			this.next.propagate();
		}
	}
	
	/**
	 * Une methode pour initialiser la forward propagation
	 */
	public void forward_init() {
		
		double[] temp = new double[this.weights[0].length];
		
		/*
		for (int k = 0 ; k < temp.length ; k++) {
			
			for (int i = 0 ; i < this.values.length ; i++) {
				
				temp[k] += this.weights[i][k]*this.values[i];
				
			}
			
			//System.out.println(temp[k]);
			
		}
		*/
		
		//System.out.println(this.next.values[0]);
		
		this.next.setValues(productMatrixVector(this.weights, this.values));		
		//this.next.setValues(temp);
		
		for (int i = 0 ; i < this.next.numberOfNeurons ; i++)
			//assert(this.next.values[i] != 0);

		this.next.propagate();		
	}
	
	/**
	 * Une methode pour obtenir le nombre de neurones presents dans la couche
	 * @return le nombre de neurones presents dans la couche
	 */
	public int getNumberOfNeurons() {		
		return this.numberOfNeurons;		
	}
	
	
	public void backprop_start(double[] expectedResult, double learningFactor) {
		
		//System.out.println("backprop_start");
		
		this.differentialErrorValues = new double[this.numberOfNeurons];
		
		//System.out.println(this.values.length);
		this.differentialErrorValues = lossFunctionDerivative(this.values, expectedResult);
		
		this.precedent.backprop_init(this.differentialErrorValues, learningFactor);
		
		}
	
	/**
	 * Une methode pour initialiser la backpropagation sur cette couche
	 * @param expectedResult le tableau des valeurs attendues a la sortie
	 * @param learningFactor le coefficient d'apprentissage pour cette backpropagation
	 */
	public void backprop_init(double[] error, double learningFactor){
		
		//System.out.println("backprop_init");
		
		/**
		 * Le tableau qui contient le jacobien de l'erreur par le vecteur de sortie
		 * voir les annexes du rapport pour plus d'informations
		 */
		
			
		double[] product = productMatrixVector(this.weights, this.values);
		double[] activatedProduct = this.activationDerivative(product);
		
		//System.out.println(activatedProduct[0]);
		//System.out.println(this.values[0]);
		this.delta = Layer.hadamartProduct(error, activatedProduct);
//		System.out.println(this.delta[0]);
		
		//System.out.println(this.weights[0][0]);
		
		this.differentialErrorWeights = Layer.productVectorVector(this.delta, this.values);
		
		this.delta = Layer.productMatrixVector(Layer.transpose(this.weights), this.delta);
		
		//System.out.println(this.differentialErrorWeights[0][0]);
		
		this.weights = (this.soustractionMatrice(this.weights, this.scalaireMatrice(learningFactor, this.differentialErrorWeights))); 
		
		//System.out.println(this.weights[0][0]);
		
		//System.out.println("Backprop initialised");
		
		//On lance l'appel de la backprop sur la couche precedente en passant en argument le vecteur que l'on vient de calculer, ainsi que le facteur d'apprentisssage qui reste le même
		if (this.precedent != null)
			this.precedent.backprop(this.delta, learningFactor);							
		
	}

	/* 
	 This methods takes as input the values coming from the last layer visited by the backpropagation algorithm
	 Note : in this method we always call the weights matrix from the layer stored in the *precedent* variable, because of the way the forward propagation is implemented
	 */
	/**
	 * Methode pour effectuer la backprop proprement dite : elle modifie les poids de la couche et est appellee recursivement sur la couche precedent si elle existe
	 * @param incomingValues le jacobien de l'erreur par les valeurs de la couche suivante dans le reseau de neurones : celle qui a appelle la methode
	 * @param learningFactor le facteur d'apprentissage qui definit l'importance de la correction sur les poids
	 */
	public void backprop(double[] incomingDelta, double learningFactor) {		
		
		/**
		 * Variable contenant le nombre de neurones dans la couche suivante
		 */
		int nextNumberOfNeurons = this.next.getNumberOfNeurons();
		/**
		 * Variable contenant la derivee de l'erreur par la matrice des poids de la couche
		 */
		this.differentialErrorWeights = new double[this.numberOfNeurons][nextNumberOfNeurons];
		/**
		 * Vecteur contenant le jacobien de l'erreur par le produit de la matrice des poids de cette couche avec le vecteur des valeurs de cette couche
		 */
		this.differentialErrorProduct = new double[this.numberOfNeurons];
		
		/**
		 * Vecteur contenant le produit de la matrice des poids de cette couche et du vecteur des valeurs de cette couche
		 */
		double[] product;
		/**
		 * Vecteur contenant le produit de la matrice des poids de cette couche et du vecteur des valeurs de cette couche, auquel on a applique la derivee de la fonction d'activation sur chacune de ses composantes
		 */
		double[] activatedProduct;
		/**
		 * Vecteur conteant le jacobien de l'erreur par les valeurs de cette couche, et qui sera passe en argument dans l'appel recursif de la methode backprop
		 */
		double[] returned = new double[this.numberOfNeurons];
		
		// On calcule les valeurs des differentes variables
		// Pour plus de precisions sur les operations en jeu, veuillez consulter l'annexe du rapport
		
		//System.out.println("activation Derivative used");
		
		product = productMatrixVector(this.weights, this.values);
		activatedProduct = this.activationDerivative(product);
		
		/*
		System.out.println("weights dimension " + Layer.transpose(this.weights).length + " " + Layer.transpose(this.weights)[0].length);
		System.out.println("incoming Delta dimension " + incomingDelta.length);
		System.out.println("activated product dimension " + activatedProduct.length);
		*/
		
		this.delta = Layer.hadamartProduct(incomingDelta, activatedProduct);
		
		
		
		this.differentialErrorWeights = Layer.productVectorVector(this.delta, this.values);
		this.weights = (this.soustractionMatrice(this.weights, this.scalaireMatrice(learningFactor, this.differentialErrorWeights))); 
		
		
		/*
		
		for(int i = 0 ; i < activatedProduct.length ; i++) {		
			this.differentialErrorProduct[i] = activatedProduct[i]*incomingValues[i];
		}
			
		this.differentialErrorWeights = productVectorVector(this.differentialErrorProduct, this.values);
		
		double[][] diffErrWeightsTest = new double[this.differentialErrorWeights.length][this.differentialErrorWeights[0].length];
				
		for (int i = 0 ; i < this.differentialErrorWeights.length ; i++) {
			
			for (int j = 0 ; j < this.differentialErrorWeights[0].length ; j++) {
				
				diffErrWeightsTest[i][j] = this.values[j]*this.differentialErrorProduct[i];
				
				//System.out.println("Values " + j + "  " + this.values[j]);
				//System.out.println("DiffErrWeightsTest " + diffErrWeightsTest[j][i]);
				//System.out.println("DiffErrorWeights " + i + " " + j + "  " + this.differentialErrorWeights[i][j]);
				
			}
			
		}
		
		this.differentialErrorWeights = diffErrWeightsTest;
		
		returned = productMatrixVector(transpose(this.weights), this.differentialErrorProduct);
		
		
		for (int k  = 0 ; k < returned.length ; k++) {
			
			double tempSum = 0;
			
			for (int i = 0 ; i < this.weights[0].length; i++) {
				
				tempSum += this.weights[i][k]*this.differentialErrorProduct[i];
				
			}
			
			returned[k] = tempSum;
			//System.out.println("Returned " + k + " " + returned[k]);
					
		}
		
		
		assert (returned.length == this.values.length);
	
		assert (this.weights.length == this.differentialErrorWeights.length);
		assert (this.weights[0].length == this.differentialErrorWeights[0].length);
		
		//System.out.println(this.weights[0][0]);
		// On met a jour la valeur des poids de cette couche
		this.weights = (this.soustractionMatrice(this.weights, this.scalaireMatrice(learningFactor, this.differentialErrorWeights))); 
		//System.out.println(this.weights[0][0]);
		
		*/
		
		// Si une couche precedente existe, alors on appelle la methode en passant en argument la variable returned et le m�me learningFactor
		if (this.precedent != null)
			// We call the method on the next layer to be processed, passing as input what we formerly calculated
			this.precedent.backprop(this.delta, learningFactor);
		
		
	}
	
	public static double[] hadamartProduct(double[] A, double[] B) {
		
		assert(A.length == B.length);
		
		double[] result = new double[A.length];
		
		for (int i = 0 ; i < A.length ; i++)
			result[i] = A[i]*B[i];
		
		return result;
		
		
	}
	
	
	/**
	 * Methode qui effectue le produit matriciel d'une matrice par un vecteur : AxV
	 * @param A matrice utilisee dans le calcul
	 * @param V vecteur utilise dans le calcul
	 * @return le produit matrice x vecteur
	 */
	public static double[] productMatrixVector(double[][] A, double[] V){
		
		// On verifie que les conditions necessaires pour effectuer le produit sont presentes
		assert(V.length == A.length);
		
		/**
		 * La variable utilisee pour sauvegarder des valeurs intermediaires dans le calcul
		 */
		double temp;
		/**
		 * La variable qui contient le resultat du produit
		 */
		double[] produit = new double[A[0].length];
		
		//Le calcul necessite l'utilisation de deux boucles
		for (int j = 0 ; j < A[0].length ; j++){			
			temp = 0;
			
			for(int k=0; k < V.length ; k++){			
				temp += A[k][j]*V[k];
				assert (!(Double.isNaN(A[k][j])));
				assert (!(Double.isNaN(V[k])));
				assert (!(Double.isNaN(temp)));			
			}
			
			produit[j] = temp;		
		}			
		return produit;
	}
	
	/**
	 * Methode qui effectue le produit entre deux matrices AxB
	 * @param A matrice a gauche du produit
	 * @param B matrice a droite du produit
	 * @return le produit de matrices
	 */
	public static double[][] productMatrixMatrix(double[][] A, double[][] B) {		
		
		/**
		 * Variable qui contient le resultat du produit, et qui sera retournee
		 */
		double[][] result = new double[B.length][A[0].length];
		
		//On verifie que les conditions necessaires au produit sont presentes
		assert (A.length == B[0].length);
		
		/**
		 * Variable utilisee pour stocker des resultats intermediaires
		 */
		double temp;
		for (int i = 0 ; i < B.length ; i++) {		
			for (int j = 0 ; j < A[0].length ; j++) {				
				temp = 0;
				
				for (int k = 0 ; k < A.length ; k++) {					
					temp += A[k][j]*B[i][k];					
				}
				result[i][j] = temp;			
			}
		}	
		return result;		
	}
	
	/**
	 * Methode qui effectue le produit entre deux vecteurs, le premier colonne et le deuxieme ligne
	 * @param A vecteur colonne passe en argument
	 * @param B vecteur ligne passe en argument
	 * @return la matrice produit des deux vecteurs
	 */
	public static double[][] productVectorVector(double[] A, double[] B) {	
		
		/**
		 * Variable qui contient le resultat qui sera retourne
		 */
		double[][] result = new double[B.length][A.length];
		
		for (int i = 0 ; i < B.length ; i++) {			
		
			for (int j = 0 ; j < A.length ; j++) {				
			
				result[i][j] = A[j]*B[i];			
		
			}	
			
		}
		return result;
	}
	
	/**
	 * Methode qui effectue la transposee d'une matrice
	 * @param M matrice a transposer
	 * @return la transposee de la matrice passee en argument
	 */
	public static double[][] transpose(double[][] M) {	
		
		/**
		 * Variable utilisee pour stocker le resultat qui sera renvoye
		 */
		double[][] result = new double[M[0].length][M.length];		
		for (int i = 0 ; i < M[0].length ; i++) {		
			for (int j = 0 ; j < M.length ; j++) {			
				result[i][j] = M[j][i];			
			}			
		}		
		return result;		
	}
	

	/**
	 * Fonction d'activation de la couche
	 * @param M vecteur de double
	 * @return le vecteur passe en argument, dont toutes les composantes x_i ont ete remplacees par F(x_i) avec F fonction d'activation
	 */

	public double[] activationFunction(double[] M){
		
		double[] result = new double[M.length];
		for(int i=0; i<M.length; i++){
			result[i] = 1/(1+Math.exp(-M[i]));		
		}	
		//System.out.println(result[0]);
		return result;	
	}
	
	/*
	public double[] activationFunction(double[] M){	
		double[] result = new double[M.length];
		for(int i=0; i<M.length; i++){
			result[i] = Math.tanh(M[i]);		
		}	
		//System.out.println(result[0]);
		return result;	
	}
	*/
	
	/**
	 * Methode qui mappe la derivee de la fonction d'activation sur le vecteur passe en argument
	 * @param input vecteur sur lequel on va effectuer le calcul
	 * @return vecteur avec les composantes remplacees par leur image par la derivee de la fonction d'activation
	 */
	
	public double[] activationDerivative(double[] input) {	
		
		int length = input.length;
		double[] result = new double[length];	
		for (int i = 0 ; i < input.length ; i++) {		
			result[i] = Math.exp(input[i])/Math.pow(1+Math.exp(input[i]), 2);
		}
		//System.out.println(result[0]);
		return result;		
	}
	
	/*
	public double[] activationDerivative(double[] input) {	
		
		int length = input.length;
		double[] result = new double[length];	
		for (int i = 0 ; i < input.length ; i++) {		
			result[i] = (1 - Math.pow(Math.tanh(input[i]),2));
		}
		//System.out.println(result[0]);
		return result;		
	}
	*/
	
	/**
	 * Methode qui a partir d'un vecteur de valeurs donnees et un vecteur de valeurs attendues, renvoie le vecteur de de l'erreur quadratique de chaque composante des vecteurs passes en entree
	 * @param input vecteur de donnees
	 * @param expected vecteur des valeurs attendues
	 * @return le vecteur de de l'erreur quadratique de chaque composante des vecteurs passes en entree
	 */
	public static double[] lossFunction(double[] input, double[] expected) {		
		int n = input.length;
		/**
		 * Variable intermediaire qui contient les valeurs du vecteur a retourner
		 */
		double[] result = new double[n];	
		for (int i = 0 ; i < n ; i++) {		
			result[i] = 0.5*Math.pow(expected[i] - input[i], 2);			
		}		
		return result;		
	}
	
	/**
	 * Methode qui calcule la derivee de la fonction de perte sur chaque composante de deux vecteurs
	 * @param input vecteur de valeurs donees
	 * @param expected vecteur de valeurs attendues
	 * @return vecteur de la derivee de la fonction de perte sur chaque composante de deux vecteurs
	 */
	public static double[] lossFunctionDerivative(double[] input, double[] expected) {	
		assert(input.length == expected.length);
		double[] result = new double[input.length];		
		for (int i = 0 ; i < input.length ; i++) {			
			result[i] = input[i]-expected[i];
			assert(!(Double.isNaN(input[i])));
			assert(!(Double.isNaN(result[i])));
		}
		//System.out.println(result[0]);
		return result;
	}

	
	/**
	 * Methode qui effectue la multiplication d'une matrice par un scalaire
	 * @param scalaire scalaire a multiplier
	 * @param matrice matrice qu'on souhaite multiplier
	 * @return matrice multipliee par un scalaire
	 */
	public double[][] scalaireMatrice(double scalaire, double[][] matrice) {		
		double[][] res = new double[matrice.length][matrice[0].length];		
		for (int i = 0 ; i < matrice.length ; i ++) {
			for (int j = 0 ; j < matrice[0].length ; j ++) {			
				res[i][j] = scalaire*matrice[i][j];				
			}			
		}
		return res;
	}
	
	
	/**
	 * Methode qui effectue la soustraction de deux matrices
	 * @param m1 matrice qui est dans la partie positive de la soustraction
	 * @param m2 matrice dans la partie negative de la soustraction
	 * @return la matrice resultat de la soustraction des deux matrices passees en entree
	 */
	public double[][] soustractionMatrice(double[][] m1, double[][] m2) {		
		double[][] res = new double[m1.length][m1[0].length];		
		for(int i = 0 ; i < m1.length ; i++) {
			for (int j = 0 ; j < m1[0].length ; j ++) {				
				res[i][j] = m1[i][j] - m2[i][j];				
			}
		}		
		return res;				
	}	
}
