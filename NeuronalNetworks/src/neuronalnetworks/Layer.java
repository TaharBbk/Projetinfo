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
	
	//Constructeur 1
	public Layer(double[] val, double[][] weight, Layer next){
		this.values = val;
		this.weights = weight;
		this.next=next;
		this.numberOfNeurons = this.values.length;
		this.next.setPrecedent(this);
	}
	
	//Constructeur 2
	public Layer(double[] val, double[][] weight){
		this.values = val;
		this.weights = weight;
		this.numberOfNeurons = this.values.length;
	}
	
	public Layer() {}
	
	public double[][] getWeights() {
		return weights;
	}

	public void setWeights(double[][] weights) {
		this.weights = weights;
	}

	public void setValues(double[] values) {
		this.values = values;
	}
	
	public void setPrecedent(Layer p) {
		this.precedent = p;
	}

	//Recuperation des valeurs des neurones de la couche
	public double[] getValues(){
		return this.values;
	}
	
	//Fonction d'activaton
	public void activate(){
		activationFunction(this.values);
	}
	
	//Foncion de propagation du reseau de neurones
	public void propagate(){
		if(this.next!=null){
			this.next.execute();
		}
		this.next.setValues(productMatrix(this.values, this.weights));
	}
	
	public int getNumberOfNeurons() {
		
		return this.numberOfNeurons;
		
	}
	
	//Execution de la forward propagation
	public void execute(){
		this.activate();
		this.propagate();
	}
	
	public void backprop_init(int[] expectedResult){
		
		this.differentialErrorValues = new double[this.numberOfNeurons];		// Initialisation of the vector based on info given when the constructor was called
		
		for (int i = 0 ; i < this.numberOfNeurons ; i++) {
			
			this.differentialErrorValues[i] = 2*(-this.values[i])*(expectedResult[i] - this.values[i]);		// Derivative of the error function with respect to Xi
			// Need to replace this part using methods to do the derivative of the error function
		}
		
		this.backprop(this.differentialErrorValues);							// We launch the backpropagation process
		
	}

	/* 
	 This methods takes as input the values coming from the last layer visited by the backpropagation algorithm
	 Note : in this method we always call the weights matrix from the layer stored in the *precedent* variable, because of the way the forward propagation is implemented
	 */
	public void backprop(double[] incomingValues) {
		
		if(this.precedent != null) {
			
			// Initialisation of the variables of the layer
			this.differentialErrorWeights = new double[this.numberOfNeurons][this.precedent.getNumberOfNeurons()];
			this.differentialErrorProduct = new double[this.numberOfNeurons];
			
			// Declatation of the variables used in this method
			double[] activatedProduct;
			double[] returned = new double[this.precedent.getNumberOfNeurons()];
			
			for (int i = 0 ; i < this.numberOfNeurons ; i++) {
				
				for (int j = 0; j < this.precedent.getNumberOfNeurons() ; j++) {
					
					this.differentialErrorProduct[i] += this.precedent.getWeights()[i][j]*this.precedent.getValues()[j];		// We calculate the differentialErrorProduct variable first as it is needed for further calculations
					
				}
			}
			
			activatedProduct = this.activationDerivative(this.differentialErrorProduct);
			
				
			for (int i = 0 ; i < this.numberOfNeurons ; i++) {	
				for (int j = 0 ; j < this.precedent.getNumberOfNeurons() ; j++) {
					
					this.differentialErrorWeights[i][j] = this.precedent.getValues()[j]*activatedProduct[i]*incomingValues[i];	// We calculate differentialErrorWeights here
					
				}
			}
			
			for (int k = 0 ; k < this.precedent.getNumberOfNeurons() ; k++) {
				for (int i = 0 ; i < this.numberOfNeurons ; i++) {
					
					returned[k] += this.precedent.getWeights()[i][k]*this.differentialErrorProduct[i];							// We calculate the input given to the next call of the backprop method
					
				}
				
				
			}
			
			this.precedent.setWeights(this.soustractionMatrice(this.precedent.getWeights(), this.scalaireMatrice(NeuronalNetworks.LEARNING_FACTOR, this.differentialErrorWeights))); // We modifiy the weights matrix according to the backprop algorithm
			
			this.precedent.backprop(returned);		// We call the method on the next layer to be processed, passing as input what we formerly calculated
			
		}
		
	}
	
	
	//Produit Matriciel entre un vecteur et une matrice
	public double[] productMatrix(double[] MB, double[][] MA){
		int ha = MA.length;
		int la = MA[0].length;
		int hb = MB.length;
		assert(la==hb);
		double[] produit = new double[ha];
		
		for(int i=0; i<ha; i++){
			int sum = 0;
			for(int h=0; h<hb; h++){
				sum+=MA[i][h]*MB[h];
			}
			produit[i]=sum;
		}
		
		return produit;
	}
	
	//Implementation de la fonction d'activation
	public void activationFunction(double[] M){
		
		for(int i=0; i<M.length; i++){
			
			M[i]=(1/(1+Math.exp(-1*M[i])));
			
		}
		
	}
	
	// Method mapping the derivative of the activation function on an input array
	public double[] activationDerivative(double[] input) {
		
		int length = input.length;
		double[] result = new double[length];
		
		for (int i = 0 ; i < input.length ; i++) {
			
			result[i] = 1/(2+Math.exp(result[i])+Math.exp(-result[i]));
			
		}
		
		return result;
		
	}
	
	// Method mapping the loss function on an array using a array of expected results
	public double[] lossFunction(double[] input, double[] expected) {
		
		int n = input.length;
		double[] result = new double[n];
		
		for (int i = 0 ; i < n ; i++) {
			
			result[i] = 0.5*Math.pow(expected[i] - input[i], 2);
			
		}
		
		return result;
		
	}
	
	// Method that multiplies a matrix by an integer scalar
	public double[][] scalaireMatrice(int scalaire, double[][] matrice) {
		
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
