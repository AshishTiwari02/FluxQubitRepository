package com.TestLearn;
import org.testng.annotations.Test;

//invocationCount method

public class invocationCountTest {

	
	@Test(invocationCount=5)
	public void sum()
	{
		int a =10;
		int b =10;
		System.out.println("total : "+(a+b));
	}
	
	
}
