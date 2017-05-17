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
		double[] val = {2,6,4};
		double[][] weight = {{1,2,3},
							 {2,3,0},
							 {5,6,4}};
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
		double[] calculatedResult = {1.65417,
									1.715879,
									1.714749};
		Assert.assertArrayEquals(underTestLayer.getValues(), calculatedResult, 0.00001);	
	}
	
	@Test
	public void testPropagate() {
		underTestLayer2.propagate();
		double[] calculatedResult = {2,6,4};
		for (int i=0; i<3; i++){
			System.out.println(underTestLayer.getValues()[i]);
		}
		System.out.println(1.7159*Math.tanh(5));
		Assert.assertArrayEquals(underTestLayer.getValues(), calculatedResult, 0.00001);
	}
	

}
