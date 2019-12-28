package com.TestLearn;
import org.testng.annotations.*;

//dependsOnMethods method 


public class testNGFeature{

	@Test
	public void loginTest()
	{
		System.out.println("Login Verification");
		//int b =10/0;
	}
	
	@Test(dependsOnMethods="loginTest")
	public void validApp()
	{
		System.out.println("Application Verification");
	}
	
	
}
