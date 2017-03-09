package neuronalnetworks;

public class Layer {

	double[] values;
	double[][] weights;
	private Layer Null;
	Layer next = Null;
	Layer precedent = Null;
	double[] differentialErrorValues;
	double[][] differentialErrorWeights;
	double[] differentialErrorProduct;
	int numberOfNeurons;
	int learningfactor = 1; // A Modifier
	
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
	
	//Foncion de propagation du reseau de neurone
	public void propagate(){
		if(this.next!=Null){
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
		
		this.differentialErrorValues = new double[this.numberOfNeurons];
		
		for (int i = 0 ; i < this.numberOfNeurons ; i++) {
			
			this.differentialErrorValues[i] = 2*(-this.values[i])*(expectedResult[i] - this.values[i]);
			
		}
		
		this.backprop(this.differentialErrorValues);
		
	}
	
	public void backprop(double[] incomingValues) {
		
		if(this.precedent != Null) {
			
			this.differentialErrorWeights = new double[this.numberOfNeurons][this.precedent.getNumberOfNeurons()];
			this.differentialErrorProduct = new double[this.numberOfNeurons];
			double[] activatedProduct;
			double[] returned = new double[this.precedent.getNumberOfNeurons()];
			
			for (int i = 0 ; i < this.numberOfNeurons ; i++) {
				
				for (int j = 0; j < this.precedent.getNumberOfNeurons() ; j++) {
					
					this.differentialErrorProduct[i] += this.precedent.getWeights()[i][j]*this.precedent.getValues()[j];
					
				}
			}
			
			activatedProduct = this.activationDerivative(this.differentialErrorProduct);
			
				
			for (int i = 0 ; i < this.numberOfNeurons ; i++) {	
				for (int j = 0 ; j < this.precedent.getNumberOfNeurons() ; j++) {
					
					this.differentialErrorWeights[i][j] = this.precedent.getValues()[j]*activatedProduct[i]*incomingValues[i];
					
				}
			}
			
			for (int k = 0 ; k < this.precedent.getNumberOfNeurons() ; k++) {
				for (int i = 0 ; i < this.numberOfNeurons ; i++) {
					
					returned[k] += this.precedent.getWeights()[i][k]*this.differentialErrorProduct[i];
					
				}
				
				
			}
			
			this.precedent.setWeights(this.soustractionMatrice(this.precedent.getWeights(), this.scalaireMatrice(this.learningfactor, this.differentialErrorWeights)));
			
			this.precedent.backprop(returned);
			
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
	
	public double[] activationDerivative(double[] input) {
		
		int length = input.length;
		double[] result = new double[length];
		
		for (int i = 0 ; i < input.length ; i++) {
			
			result[i] = 1/(2+Math.exp(result[i])+Math.exp(-result[i]));
			
		}
		
		return result;
		
	}
	
	public double[] lossFunction(double[] input, double[] expected) {
		
		int n = input.length;
		double[] result = new double[n];
		
		for (int i = 0 ; i < n ; i++) {
			
			result[i] = 0.5*Math.pow(expected[i] - input[i], 2);
			
		}
		
		return result;
		
	}
	
	public double[][] scalaireMatrice(int scalaire, double[][] matrice) {
		
		double[][] res = new double[matrice.length][matrice[0].length];
		
		for (int i = 0 ; i < matrice.length ; i ++) {
			for (int j = 0 ; j < matrice[0].length ; j ++) {
			
				res[i][j] = scalaire*matrice[i][j];
				
			}			
		}
		
		return res;
		
	}
	
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
