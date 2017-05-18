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
		double[] calculatedResult = {-0.05,
									0.05,
									0.08};
		Assert.assertArrayEquals(underTestLayer.getValues(), calculatedResult, 0.00001);	
	}
	
	@Test
	public void testPropagate() {
		underTestLayer2.propagate();
		double[] calculatedResult = {-0.0089999,0.0,0.0329999};
		Assert.assertArrayEquals(underTestLayer.getValues(), calculatedResult, 0.00001);
	}
	
	@Test
	public void testBackProp(){
		double[] incomingValues = {0.71, 0.34, 0.18};
		underTestLayer.backprop(incomingValues, 10);
		double[][] calculatedWeight = {{1,0.5,0.2},
				 			 {0.3,0.3,0},
				 			 {-0.7,0.6,-0.4}};
		for (int i=0; i<3; i++){
			Assert.assertEquals(underTestLayer.getPrecedent(), underTestLayer2);
			Assert.assertArrayEquals(underTestLayer.getWeights()[i], calculatedWeight[i], 0.00001);
			for (int j=0; j<3; j++){
				System.out.println(i+","+j+": " + underTestLayer.getWeights()[i][j]);
			}
		}
	}
	

}
