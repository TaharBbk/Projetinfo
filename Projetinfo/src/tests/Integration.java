package tests;

import java.io.IOException;
import java.util.Random;

import neuralnetworks.NeuralNetworks;


public class Integration {
	
	
	private static double xor(boolean a, boolean b) {
		
		double res = 0;
		
		if ((a && !b) || (!a && b))
			res = 1;
		
		return res;
		
	}
	
	public static void main(String[] args) throws IOException, ClassNotFoundException{
		
		 NeuralNetworks réseauTest = new NeuralNetworks();
		 
		 int nombreApprentissages = Integer.parseInt(args[0]);
		 double learningFactor = Double.parseDouble(args[1]);
		 double nombreTests = (double) Integer.parseInt(args[2]);
		 
		 boolean randA;
		 boolean randB;
		 double[] input = new double[2];
		 Random RNG = new Random();
		 double[] result;
		 double eqm = 0;
		 double successRate = 0;
		 double[] expectedOutput = new double[1];
		 double xor;
		 
		 for (double i = 1 ; i <= nombreApprentissages ; i++) {
			
			 randA = RNG.nextBoolean();
			 randB = RNG.nextBoolean();
			 
			 input[0] = (randA)? 1 : 0;
			 input[1] = (randB)? 1 : 0;
			 
			 double[] sortie = réseauTest.forwardPropagationRAM(input);
			 
			 //System.out.println(sortie[0]);
			 
			 
			 expectedOutput[0] = Integration.xor(randA, randB);
			 //expectedOutput[1] = Integration.xor(randA, randB);
			 
			 /*
			 System.out.println(input[0]);
			 System.out.println(input[1]);
			 System.out.println(expectedOutput[0]);
			 System.out.println("------------");
			 */
			 
			 //System.out.println("Démarrage @Integration");
			 réseauTest.layers[réseauTest.layers.length-1].backprop_start(expectedOutput, learningFactor);
			 
			 //réseauTest.forwardPropagationRAM(input);
			 
			 //réseauTest.layers[2].backprop_init(expectedOutput, learningFactor);
			 
			 //réseauTest.forwardPropagationRAM(input);
			 
			 //réseauTest.layers[2].backprop_init(expectedOutput, learningFactor);
			 
			 //réseauTest.forwardPropagationRAM(input);
			 
			 //réseauTest.layers[2].backprop_init(expectedOutput, learningFactor);
			 
		 
		 }
		 
		 for (int i = 0 ; i < nombreTests ; i++) {
			 
			 
			 randA = RNG.nextBoolean();
			 randB = RNG.nextBoolean();
			 
			 input[0] = (randA)? 1 : 0;
			 input[1] = (randB)? 1 : 0;
			 
			 result = réseauTest.forwardPropagationRAM(input);
			 
			 xor = Integration.xor(randA, randB);
			 
			 /*
			 System.out.println(input[0]);
			 System.out.println(input[1]);
			 System.out.println(result[0]);
			 //System.out.println(result[1]);
			*/
			 
			 
			 eqm += Math.pow((result[0]-xor), 2);
			 
			 if ((xor > 0 && result[0] > 0.5) || (xor == 0 && result[0] <= 0.5))
				 successRate += 1;
			 
		 }
		 
		 /*
		 System.out.println("0 0 " + Integration.xor(false, false));
		 System.out.println("0 1 " + Integration.xor(false, true));
		 System.out.println("1 0 " + Integration.xor(true, false));
		 System.out.println("1 1 " + Integration.xor(true, true));
		 */
		 
		 eqm /= nombreTests;
		 
		 successRate /= nombreTests;
		 
		 System.out.println("Eqm " + eqm);
		 System.out.println("Success rate " + successRate);
		
	}
	
	

}
