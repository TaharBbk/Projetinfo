package neuronalnetworks;

public class Layer {

	double[] values;						// The vector of the values of the layer
	double[][] weights;						// The matrix of the weights of the layer
	Layer next = null;						// The next layer in the NN, defaulted to null
	Layer precedent = null;					// The precedent layer in the NN
	double[] differentialErrorValues;		// The derivative of the error with respect to the values of this layer
	double[][] differentialErrorWeights;	// The derivative of the error with respect to the weights of this layer
	double[] differentialErrorProduct; 		// The derivative of the error with respect to the product of the weights of the precedent layer by the values of the precedent layer
	int numberOfNeurons;					// The number of neurons in the layer
	double activationFunctionLinearCoeff = 0.001;
	
	/**
	 * Un constructeur qui permet de cr�er une couche en prenant une matrice de poids donn�e
	 * @param val le vecteur des valeurs des neurones de la couche
	 * @param weight la matrice de poids que poss�de la couche
	 * @param next la couche suivante dans le r�seau de neurones
	 */
	public Layer(double[] val, double[][] weight, Layer next){
		this.values = val;
		this.weights = weight;
		this.next = next;
		this.numberOfNeurons = this.values.length;
		this.next.setPrecedent(this);
	}
	
	/**
	 * Un constructeur qui ne demande pas de matrice de poids � la cr�ation de la couche
	 * @param val La liste des valeurs des neurones de la couche
	 */
	public Layer(double[] val){
		this.values = val;
		this.numberOfNeurons = this.values.length;
	}
	
	/**
	 * Une m�thode pour obtenir les poids de la couche
	 * @return la matrice des poids
	 */
	public double[][] getWeights() {
		return weights;
	}

	/**
	 * Une m�thode pour changer les poids de la couche
	 * @param weights les nouveaux poids de la couche
	 */
	public void setWeights(double[][] weights) {
		this.weights = weights;
	}

	/**
	 * Une m�thode pour assigner des valeurs aux neurones de la couche
	 * @param values le vecteur des nouvelles valeurs
	 */
	public void setValues(double[] values) {
		
		this.values = values;
		//System.out.println(this.values.length);
	}
	
	/**
	 * Une m�thode pour d�finir la couche pr�c�dente dans le r�seau de neurones
	 * @param p la couche �tant � d�finir comme �tant la pr�c�dente de celle-ci
	 */
	public void setPrecedent(Layer p) {
		this.precedent = p;
	}
	
	/**
	 * Une m�thode pour obtenir la couche pr�c�dent cette couche
	 * @return la couche pr�c�dent cette couche si elle existe, null sinon
	 */
	public Layer getPrecedent() {
		return this.precedent;
	}
	
	/**
	 * Une m�thode pour d�finir la couche suivant celle-ci dans le r�seau de neurones
	 * @param n la couche � d�finir comme �tant la suivante
	 */
	public void setNext(Layer n) {
		this.next = n;
	}

	/**
	 * Une m�thode pour r�cup�rer la valeur des neurones de la couche cach�e
	 * @return la valeur des neurones de la couche
	 */
	public double[] getValues(){
		return this.values;
	}
	
	/**
	 * Une m�thode pour appliquer la fonction d'activation aux valeurs de cette couche
	 */
	public void activate(){
		this.values=activationFunction(this.values);
	}
	
	/**
	 * Une m�thode pour effectuer la forward propagation
	 */
	public void propagate(){
		
		//On v�rifie qu'aucune valeur qui a �t� entr�e est NaN
		for (int i = 0 ; i < this.values.length ; i++) {
			
			assert(!(Double.isNaN(this.values[i])));
			
			
		}
		
		//On effectue l'activation des valeurs
		this.activate();
		
		/*S'il existe une couche suivante, on modifie ses valeurs comme �tant 
		 * le r�sultat de la multiplication de la matrice des poids de cette couche
		 * avec les valeurs de cette couche*/
		if(this.next!=null){
			this.next.setValues(productMatrixVector(this.weights, this.values));
			this.next.propagate();
		}
	}
	
	/**
	 * Une m�thode pour initialiser la forward propagation
	 */
	public void forward_init() {
		
		//Il ne faut pas appliquer la fonction d'activation pour la couche d'entr�e, on passe donc directement � la transmission des valeurs
		this.next.setValues(productMatrixVector(this.weights, this.values));
		this.next.propagate();
		
	}
	
	/**
	 * Une m�thode pour obtenir le nombre de neurones pr�sents dans la couche
	 * @return le nombre de neurones pr�sents dans la couche
	 */
	public int getNumberOfNeurons() {
		
		return this.numberOfNeurons;
		
	}
	
	/**
	 * Une m�thode pour initialiser la backpropagation sur cette couche
	 * @param expectedResult le tableau des valeurs attendues � la sortie
	 * @param learningFactor le coefficient d'apprentissage pour cette backpropagation
	 */
	public void backprop_init(double[] expectedResult, double learningFactor){
		
		/**
		 * Le tableau qui contient le jacobien de l'erreur par le vecteur de sortie
		 * voir les annexes du rapport pour plus d'informations
		 */
		this.differentialErrorValues = new double[this.numberOfNeurons];
		
		//On calcule la valeur du tableau
		this.differentialErrorValues = lossFunctionDerivative(this.values, expectedResult);
		
		
		for (int i = 0 ; i < this.differentialErrorValues.length ; i++)
			assert (!(Double.isNaN(this.differentialErrorValues[i])));
		
		//On lance l'appel de la backprop sur la couche pr�c�dente en passant en argument le vecteur que l'on vient de calculer, ainsi que le facteur d'apprentisssage qui reste le m�me
		this.precedent.backprop(this.differentialErrorValues, learningFactor);						
		
	}

	/* 
	 This methods takes as input the values coming from the last layer visited by the backpropagation algorithm
	 Note : in this method we always call the weights matrix from the layer stored in the *precedent* variable, because of the way the forward propagation is implemented
	 */
	/**
	 * M�thode pour effectuer la backprop proprement dite : elle modifie les poids de la couche et est appell�e r�cursivement sur la couche pr�c�dent si elle existe
	 * @param incomingValues le jacobien de l'erreur par les valeurs de la couche suivante dans le r�seau de neurones : celle qui a appell� la m�thode
	 * @param learningFactor le facteur d'apprentissage qui d�finit l'importance de la correction sur les poids
	 */
	public void backprop(double[] incomingValues, double learningFactor) {
		
			
		/**
		 * Variable contenant le nombre de neurones dans la couche suivante
		 */
		int nextNumberOfNeurons = this.next.getNumberOfNeurons();
		/**
		 * Variable contenant le jacobien de l'erreur par la matrice des poids de la couche
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
		 * Vecteur contenant le produit de la matrice des poids de cette couche et du vecteur des valeurs de cette couche, auquel on a appliqu� la d�riv�e de la fonction d'activation sur chacune de ses composantes
		 */
		double[] activatedProduct;
		/**
		 * Vecteur conteant le jacobien de l'erreur par les valeurs de cette couche, et qui sera pass� en argument dans l'appel r�cursif de la m�thode backprop
		 */
		double[] returned = new double[this.numberOfNeurons];
		
		// On calcule les valeurs des diff�rentes variables
		// Pour plus de pr�cisions sur les op�rations en jeu, veuillez consulter l'annexe du rapport
		
		product = productMatrixVector(this.weights, this.values);
		activatedProduct = this.activationDerivative(product);
		
		for(int i = 0 ; i < activatedProduct.length ; i++) {
			
			this.differentialErrorProduct[i] = activatedProduct[i]*incomingValues[i];
			
		}
			
		this.differentialErrorWeights = productVectorVector(this.differentialErrorProduct, this.values);
		
		returned = productMatrixVector(transpose(this.weights), this.differentialErrorProduct);
		
		assert (returned.length == this.values.length);
		assert (this.weights.length == this.differentialErrorWeights.length);
		assert (this.weights[0].length == this.differentialErrorWeights[0].length);
		
		// On met � jour la valeur des poids de cette couche
		this.weights = (this.soustractionMatrice(this.weights, this.scalaireMatrice(learningFactor, this.differentialErrorWeights))); 
		
		// Si une couche pr�c�dente existe, alors on appelle la m�thode en passant en argument la variable returned et le m�me learningFactor
		if (this.precedent != null)
			this.precedent.backprop(returned, learningFactor);		// We call the method on the next layer to be processed, passing as input what we formerly calculated
		
		
	}
	
	/**
	 * M�thode qui effectue le produit matriciel d'une matrice par un vecteur : AxV
	 * @param A matrice utilis�e dans le calcul
	 * @param V vecteur utilis� dans le calcul
	 * @return le produit matrice x vecteur
	 */
	public static double[] productMatrixVector(double[][] A, double[] V){
		
		// On v�rifie que les conditions n�c�ssaires pour effectuer le produit sont pr�sentes
		assert(V.length == A.length);
		
		/**
		 * La variable utilis�e pour sauvegarder des valeurs interm�diaires dans le calcul
		 */
		double temp;
		/**
		 * La variable qui contient le r�sultat du produit
		 */
		double[] produit = new double[A[0].length];
		
		//Le calcul n�c�ssite l'utilisation de deux boucles
		for (int j = 0 ; j < A[0].length ; j++){
			
			temp = 0;
			
			for(int k=0; k < V.length ; k++){
			
				temp += A[k][j]*V[k];
			
			}
			
			produit[j] = temp;
		
		}
			
		return produit;
	}
	
	/**
	 * M�thode qui effectue le produit entre deux matrices AxB
	 * @param A matrice � gauche du produit
	 * @param B matrice � droite du produit
	 * @return le produit de matrices
	 */
	public static double[][] productMatrixMatrix(double[][] A, double[][] B) {
		
		/**
		 * Variable qui contient le r�sultat du produit, et qui sera retourn�e
		 */
		double[][] result = new double[B.length][A[0].length];
		
		//On v�rifie que les conditions n�c�ssaires au produit sont pr�sentes
		assert (A.length == B[0].length);
		
		/**
		 * Variable utilis�e pour stocker des r�sultats interm�diaires
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
	 * M�thode qui effectue le produit entre deux vecteurs, le premier colonne et le deuxieme ligne
	 * @param A vecteur colonne pass� en argument
	 * @param B vecteur ligne pass� en argument
	 * @return la matrice produit des deux vecteurs
	 */
	public static double[][] productVectorVector(double[] A, double[] B) {
		
		/**
		 * Variable qui contient le r�sultat qui sera retourn�
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
	 * M�thode qui effectue la transpos�e d'une matrice
	 * @param M matrice � transposer
	 * @return la transpos�e de la matrice pass�e en argument
	 */
	public static double[][] transpose(double[][] M) {
		
		/**
		 * Variable utilis�e pour stocker le r�sultat qui sera renvoy�
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
	 * @return le vecteur pass� en argument, dont toutes les composantes x_i ont �t� remplac�es par F(x_i) avec F fonction d'activation
	 */
	public double[] activationFunction(double[] M){
		
		/**
		 * Variable contenant le r�sultat, qui sera retourn�e
		 */
		double[] result = new double[M.length];
		
		for(int i=0; i<M.length; i++){
			
			result[i] = 1.7159*Math.tanh(2/3*M[i])+this.activationFunctionLinearCoeff*M[i];
			
		}
		
		return result;
		
	}
	
	/**
	 * M�thode qui mappe la d�riv�e de la fonction d'activation sur le vecteur pass� en argument
	 * @param input vecteur sur lequel on va effectuer le calcul
	 * @return vecteur avec les composantes remplac�es par leur image par la d�riv�e de la fonction d'activation
	 */
	public double[] activationDerivative(double[] input) {
		
		/**
		 * Variable interm�diaire qui contient les valeurs du vecteur � retourner
		 */
		double[] result = new double[input.length];
		
		for (int i = 0 ; i < input.length ; i++) {
			
			result[i] = 1.7159*(4/(3*(Math.cosh(4/3*input[i])+1)))+this.activationFunctionLinearCoeff;
			
		}
		return result;
		
	}
	
	/**
	 * M�thode qui � partir d'un vecteur de valeurs donn�es et un vecteur de valeurs attendues, renvoie le vecteur de de l'erreur quadratique de chaque composante des vecteurs pass�s en entr�e
	 * @param input vecteur de donn�es
	 * @param expected vecteur des valeurs attendues
	 * @return le vecteur de de l'erreur quadratique de chaque composante des vecteurs pass�s en entr�e
	 */
	public static double[] lossFunction(double[] input, double[] expected) {
		
		//On v�rifie que la condition n�c�ssaire est respect�e
		assert(input.length == expected.length);
		/**
		 * Variable interm�diaire qui stocke le r�sultat � retourner
		 */
		double[] result = new double[input.length];
		
		for (int i = 0 ; i < input.length ; i++) {
			
			result[i] = 0.5*Math.pow(expected[i] - input[i], 2);
			
		}
		
		return result;
		
	}
	
	/**
	 * M�thode qui calcule la d�riv�e de la fonction de perte sur chaque composante de deux vecteurs
	 * @param input vecteur de valeurs don�es
	 * @param expected vecteur de valeurs attendues
	 * @return vecteur de la d�riv�e de la fonction de perte sur chaque composante de deux vecteurs
	 */
	public static double[] lossFunctionDerivative(double[] input, double[] expected) {
		
		double[] result = new double[input.length];
		
		for (int i = 0 ; i < input.length ; i++) {
			
			result[i] = (-2)*(expected[i]-input[i]);
			
		}
		
		return result;
		
	}
	
	/**
	 * M�thode qui effectue la multiplication d'une matrice par un scalaire
	 * @param scalaire scalaire � multiplier
	 * @param matrice matrice qu'on souhaite multiplier
	 * @return matrice multipli�e par un scalaire
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
	 * M�thode qui effectue la soustraction de deux matrices
	 * @param m1 matrice qui est dans la partie positive de la soustraction
	 * @param m2 matrice dans la partie n�gative de la soustraction
	 * @return la matrice r�sultat de la soustraction des deux matrices pass�es en entr�e
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
