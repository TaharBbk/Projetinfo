package tests;

import java.io.IOException;

public class Testing {
	
	public static void main(String[] args) throws IOException, ClassNotFoundException{
		
		for (double i = 6 ; i < 7; i+=0.01) {
			
			System.out.println(i);
			Integration.main(new String[]{"10000", Double.toString(i), "100"});
			
		}
		
	}
	

}
