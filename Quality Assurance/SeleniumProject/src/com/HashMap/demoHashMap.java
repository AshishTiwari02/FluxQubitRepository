package com.HashMap;

import java.util.HashMap;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

public class demoHashMap {

	static HashMap<String,String> loginData()
	{
		HashMap <String,String> hs  = new HashMap<String,String>();
		hs.put("x", "mercury@mercury");
		hs.put("y", "mercury1@mercury1");
		hs.put("z", "mercury2@mercury2");
		
		return hs;
	}
	
	public static void main(String[] args)
	{
		String driverPath = "webdriver.chrome.driver";
		String driverURL = "D:\\Softwares\\Selenium\\chromedriver_win32\\chromedriver.exe";
		
		System.setProperty(driverPath, driverURL);
		WebDriver driver = new ChromeDriver();
		
		String cred = loginData().get("x");
		String arr[]= cred.split("@");
		
		driver.findElement(By.name("username")).sendKeys(arr[0]);
		driver.findElement(By.name("password")).sendKeys(arr[1]);
		driver.findElement(By.name("login")).click();

	}

}
