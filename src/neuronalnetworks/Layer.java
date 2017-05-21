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
	double activationFunctionLinearCoeff = 0.1;
	
	//Constructeur 1
	public Layer(double[] val, double[][] weight, Layer next){
		this.values = val;
		this.weights = weight;
		this.next = next;
		this.numberOfNeurons = this.values.length;
		this.next.setPrecedent(this);
	}
	
	//Constructeur 2
	public Layer(double[] val){
		this.values = val;
		this.numberOfNeurons = this.values.length;
	}
	
	public double[][] getWeights() {
		return weights;
	}

	public void setWeights(double[][] weights) {
		this.weights = weights;
	}

	public void setValues(double[] values) {
		
		this.values = values;
		//System.out.println(this.values.length);
	}
	
	public void setPrecedent(Layer p) {
		this.precedent = p;
	}
	
	public Layer getPrecedent() {
		return this.precedent;
	}
	
	public void setNext(Layer n) {
		this.next = n;
	}

	//Recuperation des valeurs des neurones de la couche
	public double[] getValues(){
		return this.values;
	}
	
	//Fonction d'activaton
	public void activate(){
		this.values=activationFunction(this.values);
	}
	
	//Fonction de propagation du reseau de neurones
	public void propagate(){
		if(this.next!=null){
			this.next.setValues(productMatrix(this.values, this.weights));
			this.next.execute();
		}
		else
			this.activate();
	}
	
	public void forward_init() {
		
		this.propagate();
		
	}
	
	public int getNumberOfNeurons() {
		
		return this.numberOfNeurons;
		
	}
	
	//Execution de la forward propagation
	public void execute(){
		this.activate();
		this.propagate();
	}
	
	public void backprop_init(int[] expectedResult, double learningFactor){
		this.differentialErrorValues = new double[this.numberOfNeurons];		// Initialisation of the vector based on info given when the constructor was called
		
		for (int i = 0 ; i < this.numberOfNeurons ; i++) {
			
			this.differentialErrorValues = lossFunctionDerivative(this.values, expectedResult);		// Derivative of the error function with respect to Xi
			
		}
		
		this.precedent.backprop(this.differentialErrorValues, learningFactor);							// We launch the backpropagation process
		
	}

	/* 
	 This methods takes as input the values coming from the last layer visited by the backpropagation algorithm
	 Note : in this method we always call the weights matrix from the layer stored in the *precedent* variable, because of the way the forward propagation is implemented
	 */
	public void backprop(double[] incomingValues, double learningFactor) {
		
			
		// Initialisation of the variables of the layer
		int nextNumberOfNeurons = this.next.getNumberOfNeurons();
		this.differentialErrorWeights = new double[nextNumberOfNeurons][this.numberOfNeurons];
		this.differentialErrorProduct = new double[nextNumberOfNeurons];
		
		// Declatation of the variables used in this method
		double[] activatedProduct;
		double[] returned = new double[this.getNumberOfNeurons()];
		
		for (int i = 0 ; i < nextNumberOfNeurons; i++) {
			
			for (int j = 0; j < this.numberOfNeurons ; j++) {
				
				this.differentialErrorProduct[i] += this.weights[i][j]*this.values[j];		// We calculate the differentialErrorProduct variable first as it is needed for further calculations
				
			}
		}
		
		activatedProduct = this.activationDerivative(this.differentialErrorProduct);
		
			
		for (int i = 0 ; i < nextNumberOfNeurons ; i++) {	
			for (int j = 0 ; j < this.numberOfNeurons ; j++) {
				
				this.differentialErrorWeights[i][j] = this.values[j]*activatedProduct[i]*incomingValues[i];	// We calculate differentialErrorWeights here
				
			}
		}
		
		for (int k = 0 ; k < this.numberOfNeurons ; k++) {
			for (int i = 0 ; i < nextNumberOfNeurons; i++) {
				
				returned[k] += this.weights[i][k]*this.differentialErrorProduct[i];							// We calculate the input given to the next call of the backprop method
				
			}
			
			
		}
			
		this.weights = (this.soustractionMatrice(this.weights, this.scalaireMatrice(learningFactor, this.differentialErrorWeights))); // We modifiy the weights matrix according to the backprop algorithm
		
		if (this.numberOfNeurons != 784)
			this.precedent.backprop(returned, learningFactor);		// We call the method on the next layer to be processed, passing as input what we formerly calculated
		
		
	}
	
	//Produit Matriciel entre un vecteur et une matrice
	public static double[] productMatrix(double[] MA, double[][] MB){
		int ha = MA.length;
		int hb = MB.length;
		int lb = MB[0].length;
		assert(ha==lb);
		double[] produit = new double[hb];
		
		for(int i=0; i<hb; i++){
			double sum = 0;
			for(int j=0; j<ha; j++){
				sum+=MA[j]*MB[i][j];
			}
			produit[i]=sum;
		}
			
		return produit;
	}
	
	//Implementation de la fonction d'activation
	public double[] activationFunction(double[] M){
		
		double[] result = new double[M.length];
		
		for(int i=0; i<M.length; i++){
			
			result[i] = 1.7159*Math.tanh(2/3*M[i])+this.activationFunctionLinearCoeff*M[i];
			
		}
		
		return result;
		
	}
	
	// Method mapping the derivative of the activation function on an input array
	public double[] activationDerivative(double[] input) {
		
		int length = input.length;
		double[] result = new double[length];
		
		for (int i = 0 ; i < input.length ; i++) {
			
			result[i] = 1.7159*(4/(3*(Math.cosh(4/3*input[i])+1)))+this.activationFunctionLinearCoeff;
			
		}
		return result;
		
	}
	
	// Method mapping the loss function on an array using a array of expected results
	public static double[] lossFunction(double[] input, double[] expected) {
		
		int n = input.length;
		double[] result = new double[n];
		
		for (int i = 0 ; i < n ; i++) {
			
			result[i] = 0.5*Math.pow(expected[i] - input[i], 2);
			
		}
		
		return result;
		
	}
	
	public static double[] lossFunctionDerivative(double[] input, int[] expected) {
		
		double[] result = new double[input.length];
		
		for (int i = 0 ; i < input.length ; i++) {
			
			result[i] = (-2)*(expected[i]-(double)input[i]);
			
		}
		
		return result;
		
	}
	
	// Method that multiplies a matrix by an integer scalar
	public double[][] scalaireMatrice(double scalaire, double[][] matrice) {
		
		double[][] res = new double[matrice.length][matrice[0].length];
		
		for (int i = 0 ; i < matrice.length ; i ++) {
			for (int j = 0 ; j < matrice[0].length ; j ++) {
			
				res[i][j] = scalaire*matrice[i][j];
				
			}			
		}
		
		return res;
		
	}
	
	// Method that implements substraction of matrixes : m1 - m2
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
