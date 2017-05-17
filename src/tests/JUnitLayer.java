package tests;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import neuronalnetworks.Layer;

public class JUnitLayer {
	
	Layer underTestLayer;
	Layer underTestLayer2;
	
	@Before
	public void setUp() {
		double[] val = {-0.5,0.5,0.8};
		double[][] weight = {{1,0.5,0.2},
							 {0.3,0.3,0},
							 {-0.7,0.6,-0.4}};
		underTestLayer = new Layer(val, weight);
		underTestLayer2 = new Layer(val, weight, underTestLayer);
		
	}
	
	@After
	public void tearDown() {
		underTestLayer = null;
	}
	
	@Test
	public void testActivate() {
		underTestLayer.activate();
		double[] calculatedResult = {-0.79294683,
									0.79294683,
									1.13942064};
		Assert.assertArrayEquals(underTestLayer.getValues(), calculatedResult, 0.00001);	
	}
	
	@Test
	public void testPropagate() {
		for (int i=0; i<3; i++){
			System.out.println(underTestLayer.getValues()[i]);
		}
		underTestLayer2.propagate();
		double[] calculatedResult = {-0.154015,0,0.546549};
		for (int i=0; i<3; i++){
			System.out.println(i+": " + underTestLayer.getValues()[i]);
		}
		Assert.assertArrayEquals(underTestLayer.getValues(), calculatedResult, 0.00001);
	}
	

}
